import gc
from pathlib import Path

import numpy as np
import torch
import torchvision
from ultralytics import YOLO


class UltralyticsModelWithGradients(YOLO):
    def __init__(self, model: str | Path):
        super().__init__(model)

        self.activations = {}
        self.gradients = {}
        self.concept_processor = None  # Will be set by detector

    def _save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output

        return hook

    def _save_gradient(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]

        return hook

    def register_hooks(self, bottleneck_name):
        bottleneck_layer = dict([*self.model.named_modules()])[bottleneck_name]
        if bottleneck_layer is None:
            raise ValueError(f"Layer {bottleneck_name} not found in model.")

        bottleneck_layer.register_forward_hook(self._save_activation(bottleneck_name))
        bottleneck_layer.register_full_backward_hook(
            self._save_gradient(bottleneck_name)
        )
        return

    def get_gradient(self, example, y, bottleneck_name):
        """
        Return the gradient of the loss with respect to the bottleneck activations.

        Args:
            example: input image, shape [H, W, C] or [1, H, W, C]
            y: index of the logit layer
            bottleneck_name: name of the bottleneck to get gradient wrt.
            device: torch device

        Returns:
            the gradient array (same shape as bottleneck activation).
        """
        self.register_hooks(bottleneck_name)
        self.model.eval()

        # Prepare input
        if isinstance(example, np.ndarray) and example.ndim == 3:  # noqa: F821
            example = np.expand_dims(example, axis=0)
        inputs = torch.FloatTensor(example).permute(0, 3, 1, 2).to(self.device)
        inputs.requires_grad = True

        # Forward pass (bypass Ultralytics inference_mode wrapper so grads are kept)
        with torch.enable_grad():
            # For a classification YOLO model this returns raw logits: [B, num_classes]
            logits = self.model(inputs)

        # If somehow logits packed differently, ensure it's a Tensor
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        # Compute loss on logits
        target = torch.tensor([y], dtype=torch.long, device=self.device)
        loss = torch.nn.functional.cross_entropy(logits, target)

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Get gradient w.r.t. bottleneck activations
        grads = self.gradients[bottleneck_name].detach().cpu().numpy()

        self.activations.clear()
        self.gradients.clear()
        gc.collect()

        return grads

    def run_examples(self, examples, bottleneck_name):
        self.register_hooks(bottleneck_name)
        self.model.eval()

        # Prepare input
        inputs = torch.FloatTensor(examples).permute(0, 3, 1, 2).to(self.device)

        # Forward pass (bypass Ultralytics inference_mode wrapper so grads are kept)
        with torch.enable_grad():
            _ = self.model(inputs)

        # Get bottleneck activations
        acts = self.activations[bottleneck_name]
        # If this is a detection model, take the first tensor if acts is a tuple
        if (
            hasattr(self.model, "task")
            and getattr(self.model, "task", None) == "detect"
        ):
            if isinstance(acts, tuple):
                acts = acts[0]
        acts = acts.detach()
        self.activations.clear()
        gc.collect()

        return acts

    def run_examples_cuda(self, tensor_inputs: torch.Tensor, bottleneck_name: str):
        """
        Like run_examples, but expects a torch.Tensor [B,3,H,W] already on device, no numpy conversion.
        """
        self.register_hooks(bottleneck_name)
        self.model.eval()

        # tensor_inputs: [B,3,H,W] float, already on device
        with torch.enable_grad():
            _ = self.model(tensor_inputs)

        acts = self.activations[bottleneck_name]
        if (
            hasattr(self.model, "task")
            and getattr(self.model, "task", None) == "detect"
        ):
            if isinstance(acts, tuple):
                acts = acts[0]
        acts = acts.detach()
        self.activations.clear()
        gc.collect()
        return acts

    def forward_with_features(self, image, bottleneck_layer="model.22"):
        """
        Run detection and extract bottleneck features in a single pass.

        Args:
            image: Path to image
            bottleneck_layer: Layer name to extract bottleneck features from

        Returns:
            Dictionary with detection results and per-box bottleneck features
        """
        # Store bottleneck activations
        bottleneck_features = None

        def hook_fn(module, input, output):
            nonlocal bottleneck_features
            bottleneck_features = output

        # Register hook on bottleneck layer
        layer_idx = int(bottleneck_layer.split(".")[-1])
        hook = self.model.model[layer_idx].register_forward_hook(hook_fn)

        try:
            # Run detection
            results = self(image)

            # Get detection outputs
            boxes = results[0].boxes.xyxy.cpu()
            scores = results[0].boxes.conf.cpu()
            classes = results[0].boxes.cls.cpu()

            # Skip ROI pooling if no detections
            if len(boxes) == 0:
                return {
                    "boxes": boxes,
                    "scores": scores,
                    "classes": classes,
                    "features": None,
                }

            # Skip if bottleneck features weren't captured
            if bottleneck_features is None:
                return {
                    "boxes": boxes,
                    "scores": scores,
                    "classes": classes,
                    "features": None,
                }

            # Convert to absolute coordinates if needed
            boxes_device = boxes.to(bottleneck_features.device)

            # Get image dimensions
            h_img, w_img = results[0].orig_shape

            # Calculate spatial scale for ROI Align
            # Assuming bottleneck is 1/32 of input resolution for layer 22
            h_feat, w_feat = bottleneck_features.shape[2:4]
            spatial_scale = min(h_feat / h_img, w_feat / w_img)

            # Create batch indices (0 for single image)
            batch_idx = torch.zeros(len(boxes_device), 1, device=boxes_device.device)
            rois = torch.cat([batch_idx, boxes_device], dim=1)

            # Extract per-box features using ROI Align
            roi_features = torchvision.ops.roi_align(
                bottleneck_features,
                rois,
                output_size=(1, 1),
                spatial_scale=spatial_scale,
                aligned=True,
            )

            # Flatten spatial dimensions
            roi_features = roi_features.squeeze(-1).squeeze(-1)  # [N, C]

            return {
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "features": roi_features.cpu(),
            }

        finally:
            # Remove hook
            hook.remove()

    def set_concept_processor(self, processor):
        """Set the concept processor for enhanced classification"""
        self.concept_processor = processor

    def enhance_detection_head(
        self, concept_processor, num_concept_classes=1, bottleneck_layer_idx=22
    ):
        """
        Replace the detection head with an enhanced version that outputs extended classes.
        This creates a true single-pass architecture.

        WORKAROUND: Due to NMS limitations, we temporarily keep nc=80 and store
        concept predictions separately for post-processing.
        """
        from enhanced_detection_head import ConceptEnhancedDetectionHead

        # Get the current detection layer (last layer)
        original_detect = self.model.model[-1]

        # Create enhanced detection head
        enhanced_detect = ConceptEnhancedDetectionHead(
            original_detect,
            concept_processor,
            num_concept_classes,
            bottleneck_layer_idx,
        )

        # Move enhanced detection head to the same device as the model
        enhanced_detect = enhanced_detect.to(self.device)

        # Replace the detection layer
        self.model.model[-1] = enhanced_detect
        self.enhanced_detect = enhanced_detect  # Store reference for concept retrieval

        # Install hook to capture bottleneck features during forward pass
        bottleneck_layer = self.model.model[bottleneck_layer_idx]
        self.bottleneck_hook = bottleneck_layer.register_forward_hook(
            enhanced_detect.capture_bottleneck_hook
        )

        # Keep original model metadata for NMS compatibility
        # Don't update nc - keep it at 80 for NMS

        print(
            f"✅ Enhanced detection head installed with {num_concept_classes} concept classes"
        )
        print(f"📊 Model temporarily outputs 80 classes for NMS compatibility")
        print(f"🔧 Concept predictions stored separately for post-processing")
        print(f"🎯 Bottleneck hook installed on layer {bottleneck_layer_idx}")

        return enhanced_detect

    def get_concept_predictions(self):
        """
        Retrieve stored concept predictions from the enhanced detection head.

        Returns:
            List of concept prediction tensors for each scale
        """
        # Access the enhanced detection head directly from the model's last layer
        detection_head = self.model.model[-1]
        if hasattr(detection_head, "_concept_predictions"):
            return detection_head._concept_predictions
        return None

    def forward_with_concept_enhancement(self, x):
        """
        Forward pass with concept-enhanced classification.
        Uses a simpler approach that captures bottleneck features and processes outputs post-hoc.
        """
        # Store bottleneck features during forward pass
        bottleneck_features = None

        def hook_fn(module, input, output):
            nonlocal bottleneck_features
            bottleneck_features = output

        # Register hook on bottleneck layer (model.22)
        hook = self.model.model[22].register_forward_hook(hook_fn)

        try:
            # Run normal YOLO forward pass using the model directly (not the predictor)
            with torch.no_grad():
                # Use the model's forward method directly to get raw outputs
                standard_outputs = self.model.model(x)

            # If no concept processor or no bottleneck features captured, return standard outputs
            if self.concept_processor is None or bottleneck_features is None:
                return standard_outputs

            # Process each output tensor to add concept classes
            enhanced_outputs = []

            # standard_outputs should be a list of tensors for each detection scale
            if not isinstance(standard_outputs, (list, tuple)):
                standard_outputs = [standard_outputs]

            for scale_idx, scale_output in enumerate(standard_outputs):
                if (
                    not isinstance(scale_output, torch.Tensor)
                    or len(scale_output.shape) != 3
                ):
                    enhanced_outputs.append(scale_output)
                    continue

                batch_size, num_anchors, num_attrs = scale_output.shape

                # Verify this looks like YOLO detection output (4 box coords + 1 objectness + 80 classes = 85)
                if num_attrs != 85:
                    enhanced_outputs.append(scale_output)
                    continue

                # Split into components
                box_coords = scale_output[..., :4]  # Box coordinates
                objectness = scale_output[..., 4:5]  # Objectness score
                class_scores = scale_output[..., 5:]  # Class scores (80 for COCO)

                # Enhance class scores with concept activations
                try:
                    enhanced_class_scores = (
                        self.concept_processor.enhance_classification(
                            bottleneck_features,
                            bottleneck_features,
                            class_scores,
                            scale_idx,
                        )
                    )

                    # Recombine
                    enhanced_output = torch.cat(
                        [box_coords, objectness, enhanced_class_scores], dim=-1
                    )

                    enhanced_outputs.append(enhanced_output)

                except Exception as e:
                    print(
                        f"Warning: Failed to enhance classification for scale {scale_idx}: {e}"
                    )
                    enhanced_outputs.append(scale_output)

            return enhanced_outputs

        finally:
            hook.remove()

    def _extract_raw_outputs(self, ultralytics_results):
        """
        Extract raw detection tensors from Ultralytics Results object.
        This is a simplified extraction that gets the basic detection format.
        """
        if hasattr(ultralytics_results, "__iter__") and not isinstance(
            ultralytics_results, torch.Tensor
        ):
            # It's a list of Results objects
            results = ultralytics_results[0] if ultralytics_results else None
        else:
            results = ultralytics_results

        if results is None:
            return []

        # Try to extract the detection tensor from the Results object
        if hasattr(results, "boxes") and results.boxes is not None:
            boxes = results.boxes

            # Create a simplified detection tensor in YOLO format
            # [batch, anchors, 4+1+80] where 4=coords, 1=objectness, 80=classes
            if (
                hasattr(boxes, "xyxy")
                and hasattr(boxes, "conf")
                and hasattr(boxes, "cls")
            ):
                xyxy = boxes.xyxy  # Box coordinates
                conf = boxes.conf  # Confidence scores
                cls = boxes.cls  # Class indices

                if len(xyxy) == 0:
                    # No detections
                    return [torch.zeros(1, 0, 85, device=xyxy.device)]

                # Create class scores tensor (one-hot encoding for detected classes)
                num_detections = len(xyxy)
                class_scores = torch.zeros(num_detections, 80, device=xyxy.device)

                for i, class_idx in enumerate(cls):
                    if 0 <= class_idx < 80:
                        class_scores[i, int(class_idx)] = conf[i]

                # Combine into detection format
                # Note: This is a simplified format, real YOLO outputs are more complex
                detection_tensor = torch.cat(
                    [
                        xyxy,  # [N, 4] box coordinates
                        conf.unsqueeze(1),  # [N, 1] objectness/confidence
                        class_scores,  # [N, 80] class scores
                    ],
                    dim=1,
                ).unsqueeze(0)  # Add batch dimension: [1, N, 85]

                return [detection_tensor]

        # Fallback: return empty detection
        return [torch.zeros(1, 0, 85)]

    def _enhanced_detect_forward(self, detect_layer, feature_maps, bottleneck_features):
        """
        Modified detection head that incorporates concept activations.
        """
        # Run normal detection head to get boxes and objectness
        det_output = detect_layer(feature_maps)

        # Extract box predictions and class predictions separately
        # det_output is typically [batch, anchors, 4+num_classes] for each scale

        enhanced_outputs = []

        for scale_idx, scale_output in enumerate(det_output):
            batch_size, num_anchors, num_attrs = scale_output.shape

            # Split into box coords (4), objectness (1), and class scores (80)
            box_coords = scale_output[..., :4]
            objectness = scale_output[..., 4:5]
            class_scores = scale_output[..., 5:]

            # Get corresponding feature map for this scale
            feature_map = feature_maps[scale_idx]

            # Enhance class scores with concept activations
            if self.concept_processor:
                enhanced_class_scores = self.concept_processor.enhance_classification(
                    feature_map, bottleneck_features, class_scores, scale_idx
                )
            else:
                enhanced_class_scores = class_scores

            # Recombine
            enhanced_output = torch.cat(
                [box_coords, objectness, enhanced_class_scores], dim=-1
            )

            enhanced_outputs.append(enhanced_output)

        return enhanced_outputs
