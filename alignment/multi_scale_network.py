import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAlignmentNetwork(nn.Module):
    """
    Multi-scale alignment network that aligns features from detector layers 16, 19, 22
    with classifier features. Each scale has its own projection head.
    """

    def __init__(
        self,
        detector_channels_p3=256,  # Layer 16 (P3/8)
        detector_channels_p4=512,  # Layer 19 (P4/16)
        detector_channels_p5=512,  # Layer 22 (P5/32)
        classifier_channels=512,  # Classifier features
        projection_size=256,  # Increased from 128 to 256 for better discrimination
    ):
        super().__init__()

        self.projection_size = projection_size

        # Separate projection heads for each detector scale
        self.detector_proj_p3 = nn.Sequential(
            nn.Linear(detector_channels_p3, projection_size),
            nn.ReLU(),
            nn.Linear(projection_size, projection_size),
        )

        self.detector_proj_p4 = nn.Sequential(
            nn.Linear(detector_channels_p4, projection_size),
            nn.ReLU(),
            nn.Linear(projection_size, projection_size),
        )

        self.detector_proj_p5 = nn.Sequential(
            nn.Linear(detector_channels_p5, projection_size),
            nn.ReLU(),
            nn.Linear(projection_size, projection_size),
        )

        # Single projection head for classifier features
        self.classifier_proj = nn.Sequential(
            nn.Linear(classifier_channels, projection_size),
            nn.ReLU(),
            nn.Linear(projection_size, projection_size),
        )

        # Scale fusion network to combine multi-scale features
        self.scale_fusion = nn.Sequential(
            nn.Linear(projection_size * 3, projection_size * 2),
            nn.ReLU(),
            nn.Linear(projection_size * 2, projection_size),
        )

    def forward_p3(self, detector_features):
        """Project P3 features (layer 16)"""
        projected = self.detector_proj_p3(detector_features)
        return F.normalize(projected, p=2, dim=1)

    def forward_p4(self, detector_features):
        """Project P4 features (layer 19)"""
        projected = self.detector_proj_p4(detector_features)
        return F.normalize(projected, p=2, dim=1)

    def forward_p5(self, detector_features):
        """Project P5 features (layer 22)"""
        projected = self.detector_proj_p5(detector_features)
        return F.normalize(projected, p=2, dim=1)

    def forward_classifier(self, classifier_features):
        """Project classifier features"""
        projected = self.classifier_proj(classifier_features)
        return F.normalize(projected, p=2, dim=1)

    def forward_multi_scale(self, p3_features, p4_features, p5_features):
        """
        Forward pass with multi-scale detector features.
        Combines features from all scales into a unified representation.
        """
        # Project each scale
        proj_p3 = self.detector_proj_p3(p3_features)
        proj_p4 = self.detector_proj_p4(p4_features)
        proj_p5 = self.detector_proj_p5(p5_features)

        # Concatenate and fuse
        combined = torch.cat([proj_p3, proj_p4, proj_p5], dim=1)
        fused = self.scale_fusion(combined)

        return F.normalize(fused, p=2, dim=1)

    def forward(self, detector_features, classifier_features, scale="p5"):
        """
        Standard forward pass for single-scale training compatibility.

        Args:
            detector_features: Features from detector at specified scale
            classifier_features: Features from classifier
            scale: Which scale the detector features come from ('p3', 'p4', 'p5')
        """
        if scale == "p3":
            projected_detector = self.forward_p3(detector_features)
        elif scale == "p4":
            projected_detector = self.forward_p4(detector_features)
        else:  # p5
            projected_detector = self.forward_p5(detector_features)

        projected_classifier = self.forward_classifier(classifier_features)

        return projected_detector, projected_classifier


class ScaleAwareROIPooling(nn.Module):
    """
    ROI pooling that automatically selects the best scale for each box
    based on box size and extracts features accordingly.
    """

    def __init__(self, output_size=7):
        super().__init__()
        self.output_size = output_size

    def determine_box_scales(self, boxes):
        """
        Assign each box to the most appropriate feature scale based on area.

        Args:
            boxes: Tensor of shape [N, 4] containing [x1, y1, x2, y2]

        Returns:
            List of scale names for each box
        """
        scales = []

        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height

            # Assign scale based on box area (tuned for YOLOv11 architecture)
            if area < 32 * 32:
                scales.append("p3")  # Small objects -> high resolution features
            elif area < 96 * 96:
                scales.append("p4")  # Medium objects -> medium resolution features
            else:
                scales.append("p5")  # Large objects -> low resolution features

        return scales

    def extract_roi_features(self, feature_map, box, target_size=None):
        """
        Extract ROI features from a feature map for a given bounding box.

        Args:
            feature_map: Feature map tensor [B, C, H, W]
            box: Bounding box [x1, y1, x2, y2] in image coordinates
            target_size: Target spatial size for ROI features

        Returns:
            ROI features tensor [B, C] after global average pooling
        """
        if target_size is None:
            target_size = self.output_size

        B, C, H, W = feature_map.shape
        x1, y1, x2, y2 = box

        # Convert image coordinates to feature map coordinates
        # Assuming input image is 640x640 for detector
        img_h, img_w = 640, 640

        # Scale coordinates to feature map
        fx1 = int((x1 / img_w) * W)
        fy1 = int((y1 / img_h) * H)
        fx2 = int((x2 / img_w) * W)
        fy2 = int((y2 / img_h) * H)

        # Ensure coordinates are within bounds
        fx1 = max(0, min(fx1, W - 1))
        fy1 = max(0, min(fy1, H - 1))
        fx2 = max(fx1 + 1, min(fx2, W))
        fy2 = max(fy1 + 1, min(fy2, H))

        # Extract ROI
        if fx2 > fx1 and fy2 > fy1:
            roi = feature_map[:, :, fy1:fy2, fx1:fx2]

            # Apply adaptive average pooling to get consistent size
            if target_size > 1:
                pooled_roi = F.adaptive_avg_pool2d(roi, (target_size, target_size))
                # Global average pool to get feature vector
                roi_features = torch.mean(pooled_roi, dim=[2, 3])
            else:
                # Direct global average pooling
                roi_features = torch.mean(roi, dim=[2, 3])
        else:
            # Fallback for invalid boxes
            roi_features = torch.zeros(B, C, device=feature_map.device)

        return roi_features

    def forward(self, feature_maps, boxes):
        """
        Forward pass for multi-scale ROI pooling.

        Args:
            feature_maps: Dict with keys 'p3', 'p4', 'p5' containing feature maps
            boxes: Tensor of bounding boxes [N, 4]

        Returns:
            Dict mapping box indices to extracted features and their scales
        """
        # Determine best scale for each box
        scales = self.determine_box_scales(boxes)

        # Extract features for each box using its assigned scale
        roi_features = {}
        for i, (box, scale) in enumerate(zip(boxes, scales)):
            feature_map = feature_maps[scale]
            features = self.extract_roi_features(feature_map, box)
            roi_features[i] = {"features": features, "scale": scale}

        return roi_features
