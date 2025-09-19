"""
Unified Visualization with Centralized Configuration

Merges the two visualization methods and centralizes styling configuration.
"""

import cv2
import numpy as np


class VisualizationConfig:
    """Centralized configuration for visualization styling."""

    # Modern concept colors (bright, distinctive)
    CONCEPT_COLORS = {
        "school_bus": (255, 193, 7),  # Modern amber
        "ambulance": (244, 67, 54),  # Material red
        "fire_truck": (255, 87, 34),  # Deep orange
        "tram": (33, 150, 243),  # Material blue
        "police_car": (0, 188, 212),  # Cyan 600
    }

    # Detection styling with modern colors
    COCO_COLOR = (76, 175, 80)  # Material green
    ACTIVATION_COLOR = (156, 39, 176)  # Purple 600 (for activation proposals)
    BOX_ALPHA = 0.3  # Transparency for filled boxes
    BORDER_THICKNESS = 3

    # Text styling - higher resolution
    FONT = cv2.FONT_HERSHEY_DUPLEX  # Cleaner font
    LABEL_FONT_SCALE = 0.7  # Larger for better readability
    CONCEPT_FONT_SCALE = 0.8
    TITLE_FONT_SCALE = 1.0
    TEXT_THICKNESS = 2

    # Panel styling
    PANEL_WIDTH = 450
    DETECTION_HEIGHT = 160
    PANEL_BG_COLOR = (248, 249, 250)  # Light gray
    SEPARATOR_COLOR = (218, 220, 224)  # Border gray


class LabelPositionManager:
    """Simplified label position management to avoid overlaps."""

    def __init__(self):
        self.used_positions = []

    def find_position(self, x: int, y: int, label_height: int) -> int:
        """Find non-overlapping position for label."""
        candidates = [y, y + 40, y - 40, y + 80, y - 80]

        for candidate_y in candidates:
            if not self._overlaps(x, candidate_y, label_height):
                self.used_positions.append((x, candidate_y, label_height))
                return candidate_y

        # Fallback to original position if all overlap
        self.used_positions.append((x, y, label_height))
        return y

    def _overlaps(self, x: int, y: int, height: int) -> bool:
        """Check if position overlaps with existing labels."""
        for used_x, used_y, used_height in self.used_positions:
            if abs(x - used_x) < 200 and abs(y - used_y) < (height + used_height + 10):
                return True
        return False


def visualize_detections_unified(
    image_rgb: np.ndarray,
    results: list,
    coco_names: dict[int, str],
    conf_threshold: float = 0.25,
    concept_threshold: float = 0.6,
    enhanced_mode: bool = True,
    output_path: str | None = None,
) -> np.ndarray:
    """
    Unified visualization method replacing both previous methods.

    Args:
        image_rgb: Input image in RGB format
        results: Detection results
        coco_names: COCO class names mapping
        conf_threshold: Confidence threshold for COCO detections
        concept_threshold: Confidence threshold for concept detections
        enhanced_mode: Whether to include detailed side panel
        output_path: Optional path to save result

    Returns:
        Visualization image
    """
    config = VisualizationConfig()
    position_manager = LabelPositionManager()

    # Ensure the image is contiguous for OpenCV operations
    image_rgb = np.ascontiguousarray(image_rgb)

    # Keep clean copy for cropping if enhanced mode
    clean_image = image_rgb.copy() if enhanced_mode else None

    # Process all detections
    all_concept_detections = []
    detection_id = 1

    for result in results:
        # Draw COCO detections
        _draw_coco_detections(
            image_rgb, result, coco_names, conf_threshold, config, position_manager
        )

        # Draw concept detections and collect for side panel
        concept_detections = _draw_concept_detections(
            image_rgb,
            result,
            concept_threshold,
            config,
            position_manager,
            detection_id,
            clean_image if enhanced_mode else None,
        )

        all_concept_detections.extend(concept_detections)
        detection_id += len(concept_detections)

    # Add side panel if enhanced mode
    if enhanced_mode and all_concept_detections:
        image_rgb = _add_side_panel(image_rgb, all_concept_detections, config)

    # Save if requested
    if output_path:
        output_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_image)
        print(f"Visualization saved to: {output_path}")

    return image_rgb


def _draw_coco_detections(
    image: np.ndarray,
    result,
    coco_names: dict[int, str],
    conf_threshold: float,
    config: VisualizationConfig,
    position_manager: LabelPositionManager,
):
    """Draw COCO detections on image with modern styling."""
    if not (
        hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0
    ):
        return

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, conf, cls in zip(boxes, confidences, classes):
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        class_name = coco_names[int(cls)]

        # Draw filled box with transparency
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), config.COCO_COLOR, -1)
        cv2.addWeighted(
            overlay, config.BOX_ALPHA, image, 1 - config.BOX_ALPHA, 0, image
        )

        # Draw border
        cv2.rectangle(
            image, (x1, y1), (x2, y2), config.COCO_COLOR, config.BORDER_THICKNESS
        )

        # Draw label with modern styling
        label = f"COCO: {class_name} ({conf:.3f})"
        _draw_modern_label(
            image, label, x1, y1, config.COCO_COLOR, config, position_manager
        )


def _draw_modern_label(
    image: np.ndarray,
    label: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    config: VisualizationConfig,
    position_manager: LabelPositionManager,
    is_concept: bool = False,
):
    """Draw a modern label with rounded background and high-res text."""
    font_scale = config.CONCEPT_FONT_SCALE if is_concept else config.LABEL_FONT_SCALE

    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        label, config.FONT, font_scale, config.TEXT_THICKNESS
    )

    # Find optimal position
    label_y = position_manager.find_position(x, y, text_height + 20)

    # Background with padding and rounded corners effect
    padding = 8
    bg_x1 = x
    bg_y1 = label_y - text_height - padding * 2
    bg_x2 = x + text_width + padding * 2
    bg_y2 = label_y + padding

    # Create rounded rectangle effect with multiple rectangles
    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

    # Add transparency
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)

    # Border for definition
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)

    # High-contrast text
    text_color = (255, 255, 255)  # White text for all backgrounds
    cv2.putText(
        image,
        label,
        (x + padding, label_y - padding),
        config.FONT,
        font_scale,
        text_color,
        config.TEXT_THICKNESS,
        cv2.LINE_AA,  # Anti-aliased for smooth text
    )


def _draw_concept_detections(
    image: np.ndarray,
    result,
    concept_threshold: float,
    config: VisualizationConfig,
    position_manager: LabelPositionManager,
    detection_id: int,
    clean_image: np.ndarray | None = None,
) -> list[dict]:
    """Draw concept detections and return detection info for side panel."""
    concept_detections = []

    if not hasattr(result, "concept_detections"):
        return concept_detections

    current_id = detection_id

    for concept_name, detections in result.concept_detections.items():
        for detection in detections:
            if detection["confidence"] < concept_threshold:
                continue

            box = detection["box"]
            x1, y1, x2, y2 = map(int, box)

            # Use modern styling for activation-based proposals
            if detection.get("source", "") == "activation_proposal":
                color = config.ACTIVATION_COLOR

                # Draw filled box with transparency
                overlay = image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(
                    overlay, config.BOX_ALPHA, image, 1 - config.BOX_ALPHA, 0, image
                )

                # Draw dashed border for activation proposals
                thickness = config.BORDER_THICKNESS
                dash_len = 12
                gap_len = 8

                def draw_dashed_line(
                    img, pt1, pt2, color, thickness, dash_len, gap_len
                ):
                    dist = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
                    for i in range(0, dist, dash_len + gap_len):
                        start = i
                        end = min(i + dash_len, dist)
                        if end > start:
                            x_start = int(pt1[0] + (pt2[0] - pt1[0]) * start / dist)
                            y_start = int(pt1[1] + (pt2[1] - pt1[1]) * start / dist)
                            x_end = int(pt1[0] + (pt2[0] - pt1[0]) * end / dist)
                            y_end = int(pt1[1] + (pt2[1] - pt1[1]) * end / dist)
                            cv2.line(
                                img,
                                (x_start, y_start),
                                (x_end, y_end),
                                color,
                                thickness,
                            )

                # Draw dashed rectangle
                draw_dashed_line(
                    image, (x1, y1), (x2, y1), color, thickness, dash_len, gap_len
                )  # Top
                draw_dashed_line(
                    image, (x2, y1), (x2, y2), color, thickness, dash_len, gap_len
                )  # Right
                draw_dashed_line(
                    image, (x2, y2), (x1, y2), color, thickness, dash_len, gap_len
                )  # Bottom
                draw_dashed_line(
                    image, (x1, y2), (x1, y1), color, thickness, dash_len, gap_len
                )  # Left

                # Modern detection ID badge
                badge_radius = 18
                cv2.circle(image, (x1 + 25, y1 + 25), badge_radius, color, -1)
                cv2.circle(image, (x1 + 25, y1 + 25), badge_radius, (255, 255, 255), 2)

                # ID text
                cv2.putText(
                    image,
                    str(current_id),
                    (x1 + 18, y1 + 32),
                    config.FONT,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                label = f"CONCEPT: {detection['class_name']} ({detection['confidence']:.3f}) [activation]"
                _draw_modern_label(
                    image, label, x1, y1, color, config, position_manager, True
                )
            else:
                # Regular concept detections with modern styling
                color = config.CONCEPT_COLORS.get(concept_name, (128, 128, 128))

                # Draw filled box with transparency
                overlay = image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(
                    overlay, config.BOX_ALPHA, image, 1 - config.BOX_ALPHA, 0, image
                )

                # Draw solid border
                cv2.rectangle(image, (x1, y1), (x2, y2), color, config.BORDER_THICKNESS)

                # Modern detection ID badge
                badge_radius = 18
                cv2.circle(image, (x1 + 25, y1 + 25), badge_radius, color, -1)
                cv2.circle(image, (x1 + 25, y1 + 25), badge_radius, (255, 255, 255), 2)

                # ID text
                cv2.putText(
                    image,
                    str(current_id),
                    (x1 + 18, y1 + 32),
                    config.FONT,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                label = f"CONCEPT: {detection['class_name']} ({detection['confidence']:.3f}) [{detection.get('source', '')}]"
                _draw_modern_label(
                    image, label, x1, y1, color, config, position_manager, True
                )

            # Collect for side panel
            if clean_image is not None:
                crop = _extract_detection_crop(clean_image, box)
                if detection.get("source", "") == "activation_proposal":
                    color_for_panel = config.ACTIVATION_COLOR
                else:
                    color_for_panel = config.CONCEPT_COLORS.get(
                        concept_name, (128, 128, 128)
                    )
                concept_detections.append(
                    {
                        "id": current_id,
                        "concept_name": concept_name,
                        "class_name": detection["class_name"],
                        "confidence": detection["confidence"],
                        "concept_confidence": detection["concept_confidence"],
                        "crop": crop,
                        "color": color_for_panel,
                        "reasoning": detection.get("reasoning", ""),
                        "spatial_evidence": detection.get("spatial_evidence", {}),
                    }
                )

            current_id += 1

    return concept_detections


def _extract_detection_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Extract detection crop with padding."""
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]

    # Add padding
    x1_crop = max(0, x1 - 10)
    y1_crop = max(0, y1 - 10)
    x2_crop = min(w, x2 + 10)
    y2_crop = min(h, y2 + 10)

    return image[y1_crop:y2_crop, x1_crop:x2_crop]


def _add_side_panel(
    image: np.ndarray, detections: list[dict], config: VisualizationConfig
) -> np.ndarray:
    """Add detailed side panel with detection information and modern styling."""
    panel_height = max(image.shape[0], len(detections) * config.DETECTION_HEIGHT + 60)

    # Create modern side panel
    side_panel = np.zeros((panel_height, config.PANEL_WIDTH, 3), dtype=np.uint8)
    side_panel[:, :] = config.PANEL_BG_COLOR

    # Ensure the panel is contiguous for OpenCV operations
    side_panel = np.ascontiguousarray(side_panel)

    # Modern title with better styling
    title = "CONCEPT DETECTIONS"
    title_size = cv2.getTextSize(
        title, config.FONT, config.TITLE_FONT_SCALE, config.TEXT_THICKNESS
    )[0]
    title_x = (config.PANEL_WIDTH - title_size[0]) // 2

    # Title background
    title_bg_y1 = 10
    title_bg_y2 = 50
    cv2.rectangle(
        side_panel,
        (10, title_bg_y1),
        (config.PANEL_WIDTH - 10, title_bg_y2),
        (66, 165, 245),
        -1,
    )
    cv2.rectangle(
        side_panel,
        (10, title_bg_y1),
        (config.PANEL_WIDTH - 10, title_bg_y2),
        (25, 118, 210),
        2,
    )

    # Title text
    cv2.putText(
        side_panel,
        title,
        (title_x, 35),
        config.FONT,
        config.TITLE_FONT_SCALE,
        (255, 255, 255),
        config.TEXT_THICKNESS,
        cv2.LINE_AA,
    )

    # Add detections
    y_offset = 60
    for det in detections:
        _add_detection_to_panel(side_panel, det, y_offset, config)
        y_offset += config.DETECTION_HEIGHT

    # Combine with main image
    if image.shape[0] != side_panel.shape[0]:
        # Resize to match heights
        if image.shape[0] < side_panel.shape[0]:
            scale = side_panel.shape[0] / image.shape[0]
            new_width = int(image.shape[1] * scale)
            image = cv2.resize(image, (new_width, side_panel.shape[0]))
        else:
            extension = np.zeros(
                (image.shape[0] - side_panel.shape[0], config.PANEL_WIDTH, 3),
                dtype=np.uint8,
            )
            extension[:, :] = config.PANEL_BG_COLOR
            extension = np.ascontiguousarray(extension)
            side_panel = np.vstack([side_panel, extension])

    return np.hstack([image, side_panel])


def _add_detection_to_panel(
    panel: np.ndarray, detection: dict, y_offset: int, config: VisualizationConfig
):
    """Add single detection info to side panel."""
    # Detection header
    id_text = (
        f"#{detection['id']} - {detection['concept_name'].replace('_', ' ').title()}"
    )
    cv2.putText(panel, id_text, (10, y_offset), config.FONT, 0.6, detection["color"], 2)

    # Details
    details = [
        f"Class: {detection['class_name']}",
        f"Score: {detection['confidence']:.3f}",
        f"Concept: {detection['concept_confidence']:.3f}",
    ]

    for i, detail in enumerate(details):
        cv2.putText(
            panel, detail, (10, y_offset + 25 + i * 20), config.FONT, 0.5, (0, 0, 0), 1
        )

    # Spatial evidence
    if "area" in detection["spatial_evidence"]:
        area_text = f"Area: {detection['spatial_evidence']['area']:.0f}px²"
        aspect_text = f"Aspect: {detection['spatial_evidence']['aspect_ratio']:.2f}"
        cv2.putText(
            panel, area_text, (200, y_offset + 25), config.FONT, 0.4, (60, 60, 60), 1
        )
        cv2.putText(
            panel, aspect_text, (200, y_offset + 40), config.FONT, 0.4, (60, 60, 60), 1
        )

    # Detection crop
    if detection["crop"].size > 0 and detection["crop"].shape[0] > 10:
        _add_crop_to_panel(panel, detection["crop"], detection["color"], 300, y_offset)

    # Separator line
    cv2.line(
        panel,
        (5, y_offset + 120),
        (config.PANEL_WIDTH - 5, y_offset + 120),
        config.SEPARATOR_COLOR,
        1,
    )


def _add_crop_to_panel(
    panel: np.ndarray,
    crop: np.ndarray,
    color: tuple[int, int, int],
    crop_x: int,
    crop_y: int,
):
    """Add detection crop to side panel."""
    target_size = 80
    crop_h, crop_w = crop.shape[:2]

    # Resize if needed
    if crop_h > target_size or crop_w > target_size:
        scale = min(target_size / crop_h, target_size / crop_w)
        new_w, new_h = int(crop_w * scale), int(crop_h * scale)
        crop = cv2.resize(crop, (new_w, new_h))
        crop_h, crop_w = new_h, new_w

    # Place in panel with bounds checking
    if crop_y + crop_h < panel.shape[0] and crop_x + crop_w < panel.shape[1]:
        panel[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w] = crop
        cv2.rectangle(
            panel, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), color, 2
        )
