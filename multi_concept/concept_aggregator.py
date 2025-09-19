"""
Enhanced concept aggregation for robust part-to-object detection.

This module implements improved concept detection that aggregates part-based activations
into robust object-level predictions, addressing the issue where classifier concepts
were extracted from patches and thus represent "parts" rather than whole objects.
"""

import torch
import torch.nn.functional as F


class ImprovedConceptAggregator:
    """
    Improved concept detection that aggregates part-based activations into robust object-level predictions.

    Key improvements:
    1. Spatial aggregation at multiple scales
    2. Multi-part concept fusion
    3. Density-based enhancement requiring minimum concept parts
    4. Object-level confidence pooling
    """

    def __init__(
        self, min_parts_threshold: int = 2, aggregation_scales: list[int] = [1, 2, 3]
    ):
        """
        Initialize the concept aggregator.

        Args:
            min_parts_threshold: Minimum concept parts needed for detection
            aggregation_scales: Different spatial scales for aggregation (small, medium, large objects)
        """
        self.min_parts_threshold = min_parts_threshold
        self.aggregation_scales = aggregation_scales

    def aggregate_concept_parts(
        self,
        similarity_map: torch.Tensor,  # [H, W] - concept similarity at each location
        threshold: float = 0.2,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Aggregate concept parts into robust object detections.

        Args:
            similarity_map: Spatial map of concept similarities [H, W]
            threshold: Minimum similarity for a location to be considered a "part"
            device: Device for computation

        Returns:
            aggregated_map: Enhanced similarity map with part-based aggregation [H, W]
        """
        H, W = similarity_map.shape

        # 1. Find concept "parts" (locations above threshold)
        part_mask = similarity_map > threshold

        if not part_mask.any():
            return similarity_map

        # 2. Multi-scale spatial aggregation
        aggregated_map = torch.zeros_like(similarity_map)

        # Different scales for different object sizes with emphasis on medium-scale
        scale_weights = [0.2, 0.6, 0.2]  # Small, medium, large objects

        for scale, weight in zip(self.aggregation_scales, scale_weights):
            kernel_size = 2 * scale + 1

            # Create convolution kernel for counting and summing
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
            padding = kernel_size // 2

            # Count concept parts in neighborhood
            part_count = F.conv2d(
                part_mask.float().unsqueeze(0).unsqueeze(0), kernel, padding=padding
            ).squeeze()  # [H, W]

            # Sum concept activations in neighborhood
            activation_sum = F.conv2d(
                similarity_map.unsqueeze(0).unsqueeze(0), kernel, padding=padding
            ).squeeze()  # [H, W]

            # Locations with enough parts get boosted activation
            valid_regions = part_count >= self.min_parts_threshold

            # Calculate region strength: average activation weighted by part density
            max_parts = kernel_size * kernel_size
            part_density = part_count / max_parts
            average_activation = activation_sum / (part_count + 1e-6)

            # Combined score: average activation * density boost
            region_strength = torch.where(
                valid_regions,
                average_activation * (1.0 + part_density),  # Density boost
                torch.zeros_like(activation_sum),
            )

            aggregated_map += weight * region_strength

        return aggregated_map

    def bbox_guided_aggregation(
        self,
        similarity_map: torch.Tensor,  # [H, W]
        yolo_boxes: torch.Tensor,  # [N, 4] - YOLO detected boxes in xyxy format
        feature_stride: int = 8,  # Stride from input image to feature map
        image_size: tuple[int, int] = (640, 640),  # Original image size
    ) -> list[float]:
        """
        Use YOLO's detected boxes to aggregate concept activations within object regions.

        This provides object-level concept scores by aggregating part activations
        within each detected bounding box.

        Args:
            similarity_map: Concept similarity map [H, W]
            yolo_boxes: YOLO detected bounding boxes in image coordinates [N, 4]
            feature_stride: Stride from input to feature map
            image_size: Original image size (height, width)

        Returns:
            box_concept_scores: Concept score for each detected box
        """
        if len(yolo_boxes) == 0:
            return []

        H, W = similarity_map.shape
        img_h, img_w = image_size
        box_concept_scores = []

        for box in yolo_boxes:
            x1, y1, x2, y2 = box

            # Convert to feature map coordinates
            fx1 = int((x1 / img_w) * W)
            fy1 = int((y1 / img_h) * H)
            fx2 = int((x2 / img_w) * W)
            fy2 = int((y2 / img_h) * H)

            # Clamp to feature map bounds
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(W - 1, fx2), min(H - 1, fy2)

            if fx2 > fx1 and fy2 > fy1:
                # Extract concept activations within the box
                box_similarities = similarity_map[fy1 : fy2 + 1, fx1 : fx2 + 1]

                if box_similarities.numel() > 0:
                    # Multi-metric aggregation for robust scoring
                    max_activation = torch.max(box_similarities).item()
                    mean_activation = torch.mean(box_similarities).item()

                    # Count strong activations (concept parts)
                    strong_threshold = 0.2
                    strong_parts = (box_similarities > strong_threshold).sum().item()
                    total_parts = box_similarities.numel()
                    part_density = strong_parts / total_parts if total_parts > 0 else 0

                    # Percentile-based robustness
                    sorted_activations = torch.sort(
                        box_similarities.flatten(), descending=True
                    )[0]
                    top_percentile = min(
                        max(1, int(0.1 * len(sorted_activations))),
                        len(sorted_activations),
                    )
                    top_mean = torch.mean(sorted_activations[:top_percentile]).item()

                    # Combined score with multiple factors
                    concept_score = (
                        0.3 * max_activation  # Peak concept similarity
                        + 0.3 * top_mean  # Top 10% average (robust to outliers)
                        + 0.2 * mean_activation  # Overall average
                        + 0.2 * part_density  # Density of concept parts
                    )

                    box_concept_scores.append(concept_score)
                else:
                    box_concept_scores.append(0.0)
            else:
                box_concept_scores.append(0.0)

        return box_concept_scores

    def visualize_aggregation(
        self,
        original_map: torch.Tensor,
        aggregated_map: torch.Tensor,
        save_path: str | None = None,
    ):
        """
        Visualize the concept aggregation process for debugging.

        Args:
            original_map: Original similarity map [H, W]
            aggregated_map: Aggregated similarity map [H, W]
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original similarity map
            im1 = axes[0].imshow(
                original_map.cpu().numpy(), cmap="hot", interpolation="nearest"
            )
            axes[0].set_title("Original Concept Similarities")
            axes[0].set_xlabel("Width")
            axes[0].set_ylabel("Height")
            plt.colorbar(im1, ax=axes[0])

            # Aggregated similarity map
            im2 = axes[1].imshow(
                aggregated_map.cpu().numpy(), cmap="hot", interpolation="nearest"
            )
            axes[1].set_title("Aggregated Concept Similarities")
            axes[1].set_xlabel("Width")
            axes[1].set_ylabel("Height")
            plt.colorbar(im2, ax=axes[1])

            # Difference map
            diff_map = aggregated_map - original_map
            im3 = axes[2].imshow(
                diff_map.cpu().numpy(), cmap="RdBu_r", interpolation="nearest"
            )
            axes[2].set_title("Enhancement (Aggregated - Original)")
            axes[2].set_xlabel("Width")
            axes[2].set_ylabel("Height")
            plt.colorbar(im3, ax=axes[2])

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Aggregation visualization saved to: {save_path}")

            plt.close()

        except ImportError:
            print("Matplotlib not available for visualization")

    def get_aggregation_stats(
        self,
        original_map: torch.Tensor,
        aggregated_map: torch.Tensor,
        threshold: float = 0.2,
    ) -> dict[str, float]:
        """
        Get statistics about the aggregation process.

        Args:
            original_map: Original similarity map
            aggregated_map: Aggregated similarity map
            threshold: Threshold for counting activations

        Returns:
            dictionary with aggregation statistics
        """
        stats = {}

        # Basic statistics
        stats["original_max"] = original_map.max().item()
        stats["aggregated_max"] = aggregated_map.max().item()
        stats["original_mean"] = original_map.mean().item()
        stats["aggregated_mean"] = aggregated_map.mean().item()

        # Enhancement ratio
        if stats["original_max"] > 0:
            stats["max_enhancement_ratio"] = (
                stats["aggregated_max"] / stats["original_max"]
            )
        else:
            stats["max_enhancement_ratio"] = 0.0

        # Activation counts
        stats["original_activations"] = (original_map > threshold).sum().item()
        stats["aggregated_activations"] = (aggregated_map > threshold).sum().item()

        # Concentration measure (how much aggregation concentrated the signal)
        original_entropy = self._compute_entropy(original_map)
        aggregated_entropy = self._compute_entropy(aggregated_map)
        stats["entropy_reduction"] = original_entropy - aggregated_entropy

        return stats

    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Compute spatial entropy of activation map."""
        # Normalize to probability distribution
        flat = tensor.flatten()
        flat = flat - flat.min() + 1e-8  # Ensure positive
        prob = flat / flat.sum()

        # Compute entropy
        entropy = -(prob * torch.log(prob + 1e-8)).sum().item()
        return entropy


def test_concept_aggregator():
    """Test the ImprovedConceptAggregator with synthetic data."""

    print("🔬 Testing Enhanced Concept Aggregation...")

    # Create test similarity map with scattered concept parts
    H, W = 20, 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    similarity_map = torch.zeros(H, W, device=device)

    # Add concept "parts" - scattered high similarities representing object parts
    concept_parts = [
        # Cluster 1 - strong object with multiple parts
        (5, 5, 0.8),
        (6, 5, 0.7),
        (5, 6, 0.6),
        (6, 6, 0.5),
        # Cluster 2 - medium object with fewer parts
        (15, 15, 0.5),
        (16, 15, 0.4),
        # Isolated parts - should not be enhanced much
        (2, 18, 0.3),
        (18, 2, 0.3),
    ]

    for x, y, val in concept_parts:
        similarity_map[x, y] = val

    # Test aggregation
    aggregator = ImprovedConceptAggregator(min_parts_threshold=2)
    aggregated = aggregator.aggregate_concept_parts(
        similarity_map, threshold=0.2, device=device
    )

    # Get statistics
    stats = aggregator.get_aggregation_stats(similarity_map, aggregated, threshold=0.3)

    print("📊 Aggregation Statistics:")
    print(f"   Original max similarity: {stats['original_max']:.3f}")
    print(f"   Aggregated max similarity: {stats['aggregated_max']:.3f}")
    print(f"   Enhancement ratio: {stats['max_enhancement_ratio']:.3f}")
    print(f"   Original activations > 0.3: {stats['original_activations']}")
    print(f"   Aggregated activations > 0.3: {stats['aggregated_activations']}")
    print(f"   Entropy reduction: {stats['entropy_reduction']:.3f}")

    # Test bbox-guided aggregation
    print("\n🎯 Testing Bbox-Guided Aggregation...")

    # Create mock YOLO boxes covering our concept clusters
    yolo_boxes = torch.tensor(
        [
            [40, 40, 120, 120],  # Box covering cluster 1 (scaled to image coords)
            [300, 300, 380, 380],  # Box covering cluster 2
            [20, 360, 60, 400],  # Box covering isolated part
        ],
        device=device,
        dtype=torch.float32,
    )

    box_scores = aggregator.bbox_guided_aggregation(
        aggregated, yolo_boxes, feature_stride=32, image_size=(640, 640)
    )

    print("📦 Box-level concept scores:")
    for i, score in enumerate(box_scores):
        print(f"   Box {i + 1}: {score:.3f}")

    # Save visualization if possible
    try:
        aggregator.visualize_aggregation(
            similarity_map, aggregated, save_path="concept_aggregation_test.png"
        )
    except Exception:
        print("   (Visualization skipped - matplotlib not available)")

    return stats["max_enhancement_ratio"] > 1.1  # Should enhance by at least 10%


if __name__ == "__main__":
    success = test_concept_aggregator()
    if success:
        print("\n✅ Enhanced concept aggregation working correctly!")
    else:
        print("\n❌ Enhanced concept aggregation needs debugging")
