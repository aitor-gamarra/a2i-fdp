"""
Concept-Specific Mapper with Dataclass Configuration

Maps concept activations to their specific object classes for zero-shot detection.
Uses dataclasses for cleaner, more maintainable configuration.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class SpatialRequirements:
    """Spatial constraints for object detection."""

    min_area: int = 3000
    aspect_ratio_range: tuple[float, float] = (0.8, 4.0)
    coco_classes: list[str] = field(default_factory=lambda: ["bus", "truck", "car"])


@dataclass
class ConceptRequirements:
    """Concept-based constraints for detection."""

    min_concept_confidence: float = 0.25
    min_concept_coverage: float = 0.12


@dataclass
class FusionWeights:
    """Weights for combining spatial and semantic evidence."""

    spatial: float = 0.3
    semantic: float = 0.7

    def __post_init__(self):
        """Ensure weights sum to 1.0."""
        total = self.spatial + self.semantic
        if abs(total - 1.0) > 1e-6:
            self.spatial /= total
            self.semantic /= total


@dataclass
class ConceptConfig:
    """Complete configuration for a concept-class mapping."""

    class_name: str
    description: str
    spatial_requirements: SpatialRequirements = field(
        default_factory=SpatialRequirements
    )
    concept_requirements: ConceptRequirements = field(
        default_factory=ConceptRequirements
    )
    aggregation_threshold: float = 0.15
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)


class ConceptConfigFactory:
    """Factory for creating concept configurations with defaults and overrides."""

    @staticmethod
    def create_school_bus_config() -> ConceptConfig:
        """Create configuration for school bus detection."""
        return ConceptConfig(
            class_name="school_bus",
            description="School bus vehicle",
            spatial_requirements=SpatialRequirements(coco_classes=["bus", "truck"]),
            # All other fields use defaults
        )

    @staticmethod
    def create_ambulance_config() -> ConceptConfig:
        """Create configuration for ambulance detection."""
        return ConceptConfig(
            class_name="ambulance",
            description="Ambulance emergency vehicle",
            spatial_requirements=SpatialRequirements(
                aspect_ratio_range=(1.2, 3.0), coco_classes=["truck", "car", "bus"]
            ),
            concept_requirements=ConceptRequirements(
                min_concept_confidence=0.5  # Higher threshold for ambulances
            ),
            aggregation_threshold=0.12,  # Lower threshold for more sensitive detection
            fusion_weights=FusionWeights(spatial=0.4, semantic=0.6),
        )

    @staticmethod
    def create_fire_truck_config() -> ConceptConfig:
        """Create configuration for fire truck detection."""
        return ConceptConfig(
            class_name="fire_truck",
            description="Fire truck emergency vehicle",
            spatial_requirements=SpatialRequirements(
                min_area=8000,  # Fire trucks are typically larger
                aspect_ratio_range=(1.8, 4.5),
                coco_classes=["truck", "bus"],
            ),
            concept_requirements=ConceptRequirements(
                min_concept_confidence=0.35, min_concept_coverage=0.2
            ),
            aggregation_threshold=0.2,  # Fire trucks have very distinctive features
            fusion_weights=FusionWeights(spatial=0.35, semantic=0.65),
        )

    @staticmethod
    def create_tram_config() -> ConceptConfig:
        """Create configuration for tram detection."""
        return ConceptConfig(
            class_name="tram",
            description="Tram vehicle",
            spatial_requirements=SpatialRequirements(
                aspect_ratio_range=(1.2, 3.0), coco_classes=["train"]
            ),
            aggregation_threshold=0.18,  # Trams have distinctive rail-based features
            fusion_weights=FusionWeights(spatial=0.4, semantic=0.6),
        )

    @staticmethod
    def create_police_van_config() -> ConceptConfig:
        """Create configuration for police van detection."""
        return ConceptConfig(
            class_name="police_van",
            description="Police van emergency vehicle",
            spatial_requirements=SpatialRequirements(
                min_area=2000,  # Police cars are smaller than trucks/buses
                aspect_ratio_range=(1.5, 2.8),
                coco_classes=["car", "truck"],
            ),
            concept_requirements=ConceptRequirements(
                min_concept_confidence=0.3, min_concept_coverage=0.15
            ),
            aggregation_threshold=0.16,
            fusion_weights=FusionWeights(spatial=0.45, semantic=0.55),
        )


class ConceptSpecificMapper:
    """
    Maps concept activations to their specific object classes for zero-shot detection.

    Uses dataclass-based configuration for cleaner, more maintainable setup.
    """

    def __init__(self):
        # Create configurations using factory
        self.concept_configs = {
            "school_bus": ConceptConfigFactory.create_school_bus_config(),
            "ambulance": ConceptConfigFactory.create_ambulance_config(),
            "fire_truck": ConceptConfigFactory.create_fire_truck_config(),
            "tram": ConceptConfigFactory.create_tram_config(),
            "police_van": ConceptConfigFactory.create_police_van_config(),
        }

        # Legacy compatibility: convert dataclasses to dict format
        self.concept_class_mappings = {}
        for concept_name, config in self.concept_configs.items():
            self.concept_class_mappings[concept_name] = self._config_to_dict(config)

    def _config_to_dict(self, config: ConceptConfig) -> dict:
        """Convert dataclass config to legacy dict format for backward compatibility."""
        return {
            "class_name": config.class_name,
            "description": config.description,
            "spatial_requirements": {
                "min_area": config.spatial_requirements.min_area,
                "aspect_ratio_range": config.spatial_requirements.aspect_ratio_range,
                "coco_classes": config.spatial_requirements.coco_classes,
            },
            "concept_requirements": {
                "min_concept_confidence": config.concept_requirements.min_concept_confidence,
                "min_concept_coverage": config.concept_requirements.min_concept_coverage,
            },
            "aggregation_threshold": config.aggregation_threshold,
            "fusion_weights": {
                "spatial": config.fusion_weights.spatial,
                "semantic": config.fusion_weights.semantic,
            },
        }

    def add_concept_config(self, concept_name: str, config: ConceptConfig):
        """Add a new concept configuration."""
        self.concept_configs[concept_name] = config
        self.concept_class_mappings[concept_name] = self._config_to_dict(config)

    def get_concept_config(self, concept_name: str) -> ConceptConfig:
        """Get the dataclass configuration for a concept."""
        return self.concept_configs.get(concept_name)

    def update_concept_config(self, concept_name: str, **kwargs):
        """Update specific fields of a concept configuration."""
        if concept_name not in self.concept_configs:
            raise ValueError(f"Concept '{concept_name}' not found")

        config = self.concept_configs[concept_name]

        # Update the dataclass fields
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Handle nested updates
                if key == "spatial_requirements" and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        setattr(config.spatial_requirements, sub_key, sub_value)
                elif key == "concept_requirements" and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        setattr(config.concept_requirements, sub_key, sub_value)
                elif key == "fusion_weights" and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        setattr(config.fusion_weights, sub_key, sub_value)

        # Update the legacy dict
        self.concept_class_mappings[concept_name] = self._config_to_dict(config)

    def compute_concept_specific_detections(
        self,
        concept_name: str,
        yolo_boxes: torch.Tensor,
        yolo_confidences: torch.Tensor,
        yolo_classes: torch.Tensor,
        concept_activation_map: torch.Tensor,
        image_size: tuple[int, int],
        coco_names: dict[int, str],
    ) -> list[dict]:
        """
        Compute detections for a specific concept's target class.

        Uses the concept's activation map to detect instances of that specific object type.
        """
        concept_detections = []

        if concept_name not in self.concept_configs:
            return concept_detections

        if len(yolo_boxes) == 0:
            return concept_detections

        config = self.concept_configs[concept_name]
        H, W = concept_activation_map.shape
        img_h, img_w = image_size

        # Process each YOLO detection for this specific concept
        for i, (box, yolo_conf, yolo_cls) in enumerate(
            zip(yolo_boxes, yolo_confidences, yolo_classes)
        ):
            x1, y1, x2, y2 = box.cpu().numpy()
            yolo_class_name = coco_names.get(int(yolo_cls), "unknown")

            # Check if YOLO class is compatible with this concept's target
            if yolo_class_name not in config.spatial_requirements.coco_classes:
                continue

            # Calculate spatial properties
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            aspect_ratio = box_width / max(box_height, 1e-6)

            # Check spatial requirements
            spatial_req = config.spatial_requirements
            if box_area < spatial_req.min_area or not (
                spatial_req.aspect_ratio_range[0]
                <= aspect_ratio
                <= spatial_req.aspect_ratio_range[1]
            ):
                continue

            # Map box coordinates to feature map coordinates
            fx1 = int((x1 / img_w) * W)
            fy1 = int((y1 / img_h) * H)
            fx2 = int((x2 / img_w) * W)
            fy2 = int((y2 / img_h) * H)

            # Clamp to feature map bounds
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(W - 1, fx2), min(H - 1, fy2)

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            # Extract concept activations within the bounding box
            box_concept_activations = concept_activation_map[
                fy1 : fy2 + 1, fx1 : fx2 + 1
            ]

            if box_concept_activations.numel() == 0:
                continue

            # Compute concept statistics for this specific concept
            max_concept = torch.max(box_concept_activations).item()
            mean_concept = torch.mean(box_concept_activations).item()
            concept_coverage = (box_concept_activations > 0.2).float().mean().item()
            top_percentile = torch.quantile(
                box_concept_activations.flatten(), 0.9
            ).item()

            # Combined concept confidence for this specific concept
            concept_confidence = (
                0.3 * max_concept
                + 0.3 * top_percentile
                + 0.2 * mean_concept
                + 0.2 * concept_coverage
            )

            # Check concept requirements
            concept_req = config.concept_requirements
            if (
                concept_confidence < concept_req.min_concept_confidence
                or concept_coverage < concept_req.min_concept_coverage
            ):
                continue

            # Compute fused confidence
            weights = config.fusion_weights
            fused_confidence = (
                weights.spatial * float(yolo_conf)
                + weights.semantic * concept_confidence
            )

            # Create concept-specific detection
            concept_detection = {
                "concept_name": concept_name,
                "class_name": config.class_name,
                "description": config.description,
                "confidence": fused_confidence,
                "yolo_confidence": float(yolo_conf),
                "concept_confidence": concept_confidence,
                "box": box.cpu().numpy(),
                "yolo_class": yolo_class_name,
                "spatial_evidence": {
                    "area": box_area,
                    "aspect_ratio": aspect_ratio,
                    "yolo_class": yolo_class_name,
                },
                "concept_evidence": {
                    "max_activation": max_concept,
                    "mean_activation": mean_concept,
                    "coverage": concept_coverage,
                    "top_percentile": top_percentile,
                },
                "reasoning": f"{config.class_name} detected: {yolo_class_name} ({box_area:.0f}px²) with {concept_confidence:.3f} {concept_name} concept confidence",
            }

            concept_detections.append(concept_detection)

        # Sort by fused confidence
        concept_detections.sort(key=lambda x: x["confidence"], reverse=True)
        return concept_detections


# Example usage and configuration examples
def create_custom_concept_config():
    """Example of creating a custom concept configuration."""
    custom_config = ConceptConfig(
        class_name="delivery_truck",
        description="Delivery truck vehicle",
        spatial_requirements=SpatialRequirements(
            min_area=4000, aspect_ratio_range=(1.5, 3.5), coco_classes=["truck"]
        ),
        concept_requirements=ConceptRequirements(
            min_concept_confidence=0.3, min_concept_coverage=0.15
        ),
        aggregation_threshold=0.17,
        fusion_weights=FusionWeights(spatial=0.35, semantic=0.65),
    )
    return custom_config
