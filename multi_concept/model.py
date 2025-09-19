"""
Multi-Concept Zero-Shot YOLO Model

Main implementation that uses multiple concept vectors to detect their
corresponding specific object classes.
"""

import pickle

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .detect import MultiConceptGuidedDetect
from .results import MultiConceptSemanticResults
from .visualization import visualize_detections_unified


class MultiConceptZeroShotYOLO:
    def save_concept_heatmaps(self, results, output_dir):
        """
        For each result and concept, save a heatmap visualizing the activation map over the image.
        """
        import os

        os.makedirs(output_dir, exist_ok=True)
        for result in results:
            img = result.orig_img
            img_name = os.path.splitext(os.path.basename(result.path))[0]
            if not hasattr(self.multi_concept_detect, "last_concept_activations"):
                continue
            for (
                concept_name,
                activation_map,
            ) in self.multi_concept_detect.last_concept_activations.items():
                # Resize activation map to image size if needed
                h, w = img.shape[0], img.shape[1]
                if activation_map.shape[0] != h or activation_map.shape[1] != w:
                    if isinstance(activation_map, np.ndarray):
                        tensor_map = torch.from_numpy(activation_map)[
                            None, None, ...
                        ].float()
                    else:
                        tensor_map = activation_map[None, None, ...].float()
                    activation_map_resized = np.array(
                        torch.nn.functional.interpolate(
                            tensor_map,
                            size=(h, w),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0].cpu()
                    )
                else:
                    if isinstance(activation_map, torch.Tensor):
                        activation_map_resized = activation_map.cpu().numpy()
                    else:
                        activation_map_resized = activation_map
                # Normalize activation map to [0, 1]
                act_norm = (activation_map_resized - activation_map_resized.min()) / (
                    activation_map_resized.ptp() + 1e-8
                )
                # Convert to heatmap
                heatmap = np.uint8(255 * act_norm)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # Overlay heatmap on image
                overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
                out_path = os.path.join(
                    output_dir, f"{img_name}_{concept_name}_heatmap.png"
                )
                cv2.imwrite(out_path, overlay)
                print(f"Saved heatmap for {concept_name} to {out_path}")

    """
    Multi-Concept Zero-Shot YOLO implementation that uses multiple concept vectors
    to detect their corresponding specific object classes.

    Key Features:
    - school_bus concepts → detect school buses
    - ambulance concepts → detect ambulances
    - Extensible to additional concept-class pairs
    """

    def __init__(
        self,
        yolo_model_path: str,
        alignment_model_path: str,
        concept_vectors_paths: dict[str, str],  # concept_name -> path
        device: str = "cuda",
    ):
        self.device = device

        # Load YOLO model
        print("Loading YOLO model...")
        self.original_yolo = YOLO(yolo_model_path)
        self.original_yolo.eval()
        self.model = self.original_yolo.model.to(device)
        self.model.eval()

        # Load alignment network
        print("Loading alignment network...")
        alignment_state = torch.load(alignment_model_path, map_location=device)

        from alignment.multi_scale_network import MultiScaleAlignmentNetwork

        self.alignment_net = MultiScaleAlignmentNetwork(
            detector_channels_p3=128,
            detector_channels_p4=256,
            detector_channels_p5=512,
            classifier_channels=512,
            projection_size=512,
        )
        self.alignment_net.load_state_dict(alignment_state)
        self.alignment_net.to(device)
        self.alignment_net.eval()

        # Load multiple concept vectors
        print("Loading concept vectors...")
        self.concept_vectors_dict = {}
        self.concept_names = []

        for concept_name, concept_path in concept_vectors_paths.items():
            print(f"  Loading {concept_name} from {concept_path}")
            with open(concept_path, "rb") as f:
                concept_data = pickle.load(f)

            # Handle different key formats (school_bus vs "school bus")
            concept_key = None
            if concept_name in concept_data:
                concept_key = concept_name
            elif concept_name.replace("_", " ") in concept_data:
                concept_key = concept_name.replace("_", " ")
            elif concept_name.replace(" ", "_") in concept_data:
                concept_key = concept_name.replace(" ", "_")

            if concept_key and concept_key in concept_data:
                centers = concept_data[concept_key]["concept_centers"].to(device)
                tcav_scores = torch.mean(
                    torch.FloatTensor(
                        [
                            concept_data[concept_key]["tcav_scores"][concept]
                            for concept in concept_data[concept_key]["concepts"]
                        ]
                    ).to(device),
                    dim=-1,
                )
                self.concept_vectors_dict[concept_name] = centers

                # self.concept_vectors_dict[concept_name] = F.normalize(
                #     (tcav_scores.reshape(-1, 1) * centers)[:2], dim=-1
                # )
                self.concept_names.append(concept_name)
                print(
                    f"    ✓ Loaded {concept_name} concept vectors (key: {concept_key})"
                )
            else:
                print(f"    ⚠ {concept_name} not found in {concept_path}")
                print(f"      Available keys: {list(concept_data.keys())}")

        # Replace detection head with multi-concept version
        print("Creating multi-concept detection head...")
        original_detect = self.model.model[-1]

        self.multi_concept_detect = MultiConceptGuidedDetect(
            original_detect,
            self.alignment_net,
            self.concept_vectors_dict,
            self.concept_names,
        )

        self.multi_concept_detect.eval()

        # Replace in model
        self.model.model[-1] = self.multi_concept_detect

        # CRITICAL: Update the original YOLO instance to use our modified model
        self.original_yolo.model = self.model

        # Store YOLO names
        self.coco_names = {
            i: name for i, name in enumerate(self.original_yolo.names.values())
        }

        print("✅ Multi-Concept Zero-Shot YOLO ready!")
        print("   COCO classes (spatial): 80")
        print(f"   Concept classes (semantic): {self.concept_names}")
        print(f"   Zero-shot detection for: {', '.join(self.concept_names)}")
        print("   🚀 Ready for concept-specific object detection!")

    def _propose_boxes_from_activation(
        self, activation_map, threshold=0.5, min_area=20
    ):
        """
        Propose bounding boxes from a concept activation map using only NumPy.
        Returns list of (x1, y1, x2, y2, score)
        """
        mask = activation_map > threshold
        visited = np.zeros_like(mask, dtype=bool)
        boxes = []

        def flood_fill(x, y):
            coords = [(x, y)]
            region = []
            while coords:
                cx, cy = coords.pop()
                if (
                    0 <= cx < mask.shape[0]
                    and 0 <= cy < mask.shape[1]
                    and mask[cx, cy]
                    and not visited[cx, cy]
                ):
                    visited[cx, cy] = True
                    region.append((cx, cy))
                    coords.extend(
                        [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
                    )
            return region

        # 1. Get initial boxes from connected components
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and not visited[i, j]:
                    region = flood_fill(i, j)
                    if len(region) >= min_area:
                        xs, ys = zip(*region)
                        x1, y1 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)
                        score = float(np.mean(activation_map[x1 : x2 + 1, y1 : y2 + 1]))
                        boxes.append([y1, x1, y2, x2, score])

        # 2. Merge nearby/overlapping boxes
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA + 1)
            interH = max(0, yB - yA + 1)
            interArea = interW * interH
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou

        # Much stricter merging and filtering
        min_area = max(min_area, 100)  # force minimum area to 100
        iou_thresh = 0.5
        proximity_thresh = 30

        # --- Merging logic ---
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA + 1)
            interH = max(0, yB - yA + 1)
            interArea = interW * interH
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou

        merged = []
        used = set()
        for i, boxA in enumerate(boxes):
            if i in used:
                continue
            group = [boxA]
            used.add(i)
            for j, boxB in enumerate(boxes):
                if j == i or j in used:
                    continue
                # Check IoU or proximity
                if iou(boxA, boxB) > iou_thresh or (
                    abs(boxA[0] - boxB[2]) < proximity_thresh
                    or abs(boxA[2] - boxB[0]) < proximity_thresh
                    or abs(boxA[1] - boxB[3]) < proximity_thresh
                    or abs(boxA[3] - boxB[1]) < proximity_thresh
                ):
                    group.append(boxB)
                    used.add(j)
            ys = [b[0] for b in group] + [b[2] for b in group]
            xs = [b[1] for b in group] + [b[3] for b in group]
            y1, x1 = min(ys), min(xs)
            y2, x2 = max(ys), max(xs)
            score = np.mean([b[4] for b in group])
            area = (y2 - y1 + 1) * (x2 - x1 + 1)
            if area >= min_area:
                merged.append([y1, x1, y2, x2, score])

        # --- NMS logic ---
        def nms(boxes, iou_threshold=0.5):
            if not boxes:
                return []
            boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
            keep = []
            while boxes:
                curr = boxes.pop(0)
                keep.append(curr)
                boxes = [b for b in boxes if iou(curr, b) < iou_threshold]
            return keep

        nms_boxes = nms(merged, iou_threshold=0.5)
        # --- Top-K filtering ---
        N = 3
        nms_boxes = sorted(nms_boxes, key=lambda x: x[4], reverse=True)[:N]
        return [tuple(b) for b in nms_boxes]

    def predict_concept_specific(
        self,
        source,
        conf_threshold=0.25,
        concept_activation_threshold=0.5,
        concept_min_area=20,
        **kwargs,
    ):
        """
        Predict with concept-specific zero-shot detection.

        Each concept vector detects its corresponding object class.
        """

        # ...existing code...
        self.multi_concept_detect.last_concept_activations = {}
        self.multi_concept_detect.last_backbone_features = None

        captured_features = []

        def capture_hook(layer_idx):
            def hook(module, input, output):
                captured_features.append(output.detach().clone())

            return hook

        backbone_layers = [16, 19, 22]
        hooks = []
        for i, layer_idx in enumerate(backbone_layers):
            hook = self.model.model[layer_idx].register_forward_hook(capture_hook(i))
            hooks.append(hook)

        try:
            original_forward = self.multi_concept_detect.forward

            def enhanced_forward(x_list):
                if len(captured_features) >= 3:
                    features_to_use = captured_features[-3:]
                    features_copy = [f.clone() for f in features_to_use]
                    result = original_forward(x_list, features_copy)
                else:
                    result = original_forward(x_list, None)
                return result

            self.multi_concept_detect.forward = enhanced_forward
            captured_features.clear()
            results = self.original_yolo.predict(
                source, conf=conf_threshold, verbose=False, **kwargs
            )
            self.multi_concept_detect.forward = original_forward

            enhanced_results = []
            for result in results:
                multi_result = MultiConceptSemanticResults(
                    orig_img=result.orig_img,
                    path=result.path,
                    names=result.names,
                    boxes=result.boxes.data if result.boxes is not None else None,
                )
                multi_result.boxes = result.boxes
                multi_result.orig_shape = result.orig_shape
                multi_result.speed = result.speed

                concept_detections = {}
                concept_activation_maps = {}
                for concept_name in self.concept_names:
                    concept_map = None
                    if (
                        hasattr(self.multi_concept_detect, "last_concept_activations")
                        and concept_name
                        in self.multi_concept_detect.last_concept_activations
                    ):
                        concept_map = (
                            self.multi_concept_detect.last_concept_activations[
                                concept_name
                            ]
                        )
                        concept_activation_maps[concept_name] = concept_map

                    detections = []
                    # 1. YOLO-based concept detections
                    if result.boxes is not None and len(result.boxes) > 0:
                        detections += self.multi_concept_detect.concept_mapper.compute_concept_specific_detections(
                            concept_name,
                            result.boxes.xyxy,
                            result.boxes.conf,
                            result.boxes.cls,
                            concept_map,
                            (result.orig_img.shape[0], result.orig_img.shape[1]),
                            self.coco_names,
                        )
                    # 2. Activation-based proposals (always run if concept_map exists)
                    if concept_map is not None:
                        h, w = result.orig_img.shape[0], result.orig_img.shape[1]
                        if concept_map.shape[0] != h or concept_map.shape[1] != w:
                            # If concept_map is a numpy array, convert to tensor; if already tensor, use directly
                            if isinstance(concept_map, np.ndarray):
                                tensor_map = torch.from_numpy(concept_map)[
                                    None, None, ...
                                ].float()
                            else:
                                tensor_map = concept_map[None, None, ...].float()
                            concept_map_resized = np.array(
                                torch.nn.functional.interpolate(
                                    tensor_map,
                                    size=(h, w),
                                    mode="bilinear",
                                    align_corners=False,
                                )[0, 0].cpu()
                            )
                        else:
                            # If tensor, convert to numpy for downstream use
                            if isinstance(concept_map, torch.Tensor):
                                concept_map_resized = concept_map.cpu().numpy()
                            else:
                                concept_map_resized = concept_map
                        proposed_boxes = self._propose_boxes_from_activation(
                            concept_map_resized,
                            threshold=concept_activation_threshold,
                            min_area=concept_min_area,
                        )
                        print(
                            f"[DEBUG] Activation-based proposals for concept '{concept_name}': {len(proposed_boxes)} boxes"
                        )
                        for box in proposed_boxes:
                            x1, y1, x2, y2, score = box
                            print(f"  Box: ({x1}, {y1}, {x2}, {y2}), score={score:.3f}")
                            detections.append(
                                {
                                    "xyxy": [x1, y1, x2, y2],
                                    "box": [x1, y1, x2, y2],
                                    "confidence": score,
                                    "cls": None,
                                    "concept": concept_name,
                                    "source": "activation_proposal",
                                    "class_name": concept_name,
                                    "concept_confidence": score,
                                }
                            )
                    concept_detections[concept_name] = detections
                multi_result.concept_detections = concept_detections
                multi_result.concept_activation_maps = concept_activation_maps
                enhanced_results.append(multi_result)
            return enhanced_results
        finally:
            try:
                self.multi_concept_detect.forward = original_forward
            except Exception:
                pass
            self.multi_concept_detect.last_concept_activations = {}
            self.multi_concept_detect.last_backbone_features = None
            for hook in hooks:
                try:
                    hook.remove()
                except Exception:
                    pass
            captured_features.clear()

    def visualize_concept_detections(
        self,
        source,
        results=None,
        output_path=None,
        conf_threshold=0.25,
        concept_threshold=0.48,
        enhanced_mode=True,
    ):
        """
        Unified visualization method for COCO detections and concept-specific zero-shot detections.

        Args:
            source: Image source (path or array)
            results: Optional pre-computed results, if None will run prediction
            output_path: Optional path to save result
            conf_threshold: Confidence threshold for COCO detections
            concept_threshold: Confidence threshold for concept detections
            enhanced_mode: Whether to include detailed side panel

        Returns:
            Visualization image array
        """

        # Get results if not provided
        if results is None:
            results = self.predict_concept_specific(
                source, conf_threshold=conf_threshold
            )

        # Load image
        if isinstance(source, str):
            image = cv2.imread(source)
        else:
            image = source

        if image is None:
            print(f"Could not load image: {source}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)

        # Use the unified visualization function
        result_image = visualize_detections_unified(
            image_rgb=image_rgb,
            results=results,
            coco_names=self.coco_names,
            conf_threshold=conf_threshold,
            concept_threshold=concept_threshold,
            enhanced_mode=enhanced_mode,
            output_path=output_path,
        )

        # Print mode information if saved
        if output_path:
            mode_text = "Enhanced" if enhanced_mode else "Standard"
            print(f"{mode_text} visualization saved to: {output_path}")

        return result_image
