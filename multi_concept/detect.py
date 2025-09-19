"""
Multi-Concept Guided Detection Head

Multi-concept YOLO Detect layer that processes multiple concept vectors
to detect their corresponding object classes.
"""

import torch
import torch.nn.functional as F
from ultralytics.nn.modules import Detect

from .concept_aggregator import ImprovedConceptAggregator
from .mapper import ConceptSpecificMapper


class MultiConceptGuidedDetect(Detect):
    """
    Multi-concept YOLO Detect layer that processes multiple concept vectors
    to detect their corresponding object classes.
    """

    def __init__(
        self, original_detect, alignment_net, concept_vectors_dict, concept_names
    ):
        # Extract channel configuration from original detect
        ch = tuple(
            [
                list(original_detect.cv2[i])[0].conv.in_channels
                for i in range(len(original_detect.cv2))
            ]
        )

        # Keep original 80 classes for YOLO spatial reasoning
        super().__init__(nc=80, ch=ch)

        # Copy all necessary attributes from original detect
        for attr in ["stride", "anchors", "f", "i", "inplace", "cv2", "cv3", "dfl"]:
            if hasattr(original_detect, attr):
                setattr(self, attr, getattr(original_detect, attr))

        # Multi-concept processing components
        self.alignment_net = alignment_net
        self.concept_vectors_dict = concept_vectors_dict  # concept_name -> vectors
        self.concept_names = concept_names

        # Enhanced concept aggregation
        self.concept_aggregator = ImprovedConceptAggregator(
            min_parts_threshold=2, aggregation_scales=[1, 2, 3]
        )

        # Concept-specific mapper
        self.concept_mapper = ConceptSpecificMapper()

        # Store for post-processing - one activation map per concept
        self.last_concept_activations = {}  # concept_name -> activation_map
        self.last_backbone_features = None

    def forward(self, x_list, backbone_features=None):
        """Forward pass with multi-concept activation computation."""

        # Standard YOLO forward pass - unchanged spatial reasoning
        y = []
        for i, (cv2_layer, cv3_layer) in enumerate(zip(self.cv2, self.cv3)):
            box_preds = cv2_layer(x_list[i])
            class_preds = cv3_layer(x_list[i])
            y.append(torch.cat([box_preds, class_preds], 1))

        # Compute concept activations for each concept
        if backbone_features is not None:
            self.last_concept_activations = self._compute_multi_concept_activations(
                backbone_features
            )

        self.last_backbone_features = backbone_features

        return self._inference(y) if not self.training else y

    def _compute_multi_concept_activations(self, backbone_features_list):
        """Compute activation maps for each concept separately."""

        concept_activation_maps = {}

        # Process each concept separately
        for concept_name in self.concept_names:
            if concept_name not in self.concept_vectors_dict:
                continue

            concept_vectors = self.concept_vectors_dict[concept_name]

            # Get concept-specific configuration
            concept_config = self.concept_mapper.concept_class_mappings.get(
                concept_name, {}
            )
            aggregation_threshold = concept_config.get(
                "aggregation_threshold", 0.15
            )  # Default fallback

            best_activation_map = None
            best_activation_strength = 0

            # Process each scale for this concept
            for scale_idx, backbone_features in enumerate(backbone_features_list):
                if scale_idx >= 3:  # Only process P3, P4, P5
                    break

                B, C, H, W = backbone_features.shape
                device = backbone_features.device

                # Project backbone features using alignment network
                features_flat = (
                    backbone_features.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)
                )

                # Use appropriate projection head for each scale
                with torch.no_grad():  # Ensure no gradients are computed
                    if scale_idx == 0:
                        projected = self.alignment_net.detector_proj_p3(features_flat)
                    elif scale_idx == 1:
                        projected = self.alignment_net.detector_proj_p4(features_flat)
                    else:
                        projected = self.alignment_net.detector_proj_p5(features_flat)

                    # projected = F.normalize(projected, dim=-1)

                    # Compute similarities for this specific concept
                    concept_centers = concept_vectors
                    similarities = torch.mm(projected, concept_centers.T)
                    max_similarities = torch.max(similarities, dim=1)[0]

                    # Reshape to spatial map
                    similarity_map = max_similarities.view(B, H, W)

                    # Apply enhanced aggregation for each batch
                    for b in range(B):
                        batch_map = similarity_map[b]

                        # Apply concept aggregation
                        aggregated_map = self.concept_aggregator.aggregate_concept_parts(
                            batch_map,
                            threshold=aggregation_threshold,  # Use concept-specific threshold
                            device=device,
                        )

                        # Track the scale with strongest activation for this concept
                        activation_strength = aggregated_map.max().item()
                        if activation_strength > best_activation_strength:
                            best_activation_strength = activation_strength

                            # Resize to common size
                            target_size = 20
                            best_activation_map = (
                                F.interpolate(
                                    aggregated_map.unsqueeze(0).unsqueeze(0),
                                    size=(target_size, target_size),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                .squeeze()
                                .detach()
                            )  # Detach to prevent gradient accumulation

            # Store the best activation map for this concept
            if best_activation_map is not None:
                concept_activation_maps[concept_name] = best_activation_map
            else:
                concept_activation_maps[concept_name] = torch.zeros(
                    20, 20, device=backbone_features_list[0].device
                )

        return concept_activation_maps
