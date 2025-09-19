import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from alignment.multi_scale_network import MultiScaleAlignmentNetwork
from data.imagenet_1k.imagenet_1k import Imagenet1k
from ultralytics_model_with_gradients import UltralyticsModelWithGradients


class MultiScaleAlignmentDataset(Dataset):
    """
    Dataset for training multi-scale alignment networks.
    Provides random crops from an image dataset, extracting multi-scale features
    from detector layers 16, 19, 22 and matching with classifier features.
    """

    def __init__(
        self,
        dataset,
        detector_model_path: str,
        classifier_model_path: str,
        detector_input_size=(640, 640),
        classifier_input_size=(224, 224),
        num_crops_per_image=3,
        device: str = "cuda",
        max_boxes_per_image: int = 10,
    ):
        self.dataset = dataset
        self.detector_input_size = detector_input_size
        self.classifier_input_size = classifier_input_size
        self.num_crops_per_image = num_crops_per_image
        self.device = device
        self.max_boxes_per_image = max_boxes_per_image

        # Load detector and classifier models
        self.detector = UltralyticsModelWithGradients(detector_model_path)
        self.detector.eval()
        self.detector.to(device)

        self.classifier = UltralyticsModelWithGradients(classifier_model_path)
        self.classifier.eval()
        self.classifier.to(device)

        # Define transform to get random crops
        self.crop_transform = transforms.RandomResizedCrop(
            size=max(detector_input_size),  # Crop to a size that can be downscaled
            scale=(0.2, 1.0),  # Crop 20% to 100% of the image
            ratio=(0.75, 1.33),
        )

        print(
            f"Multi-scale alignment dataset initialized with {len(dataset)} base images"
        )

    def __len__(self):
        return len(self.dataset) * self.num_crops_per_image

    def __getitem__(self, idx):
        """Get a training sample with multi-scale detector features and classifier features."""
        original_idx = int(idx // self.num_crops_per_image)
        item = self.dataset[original_idx]
        original_image = item["image"]

        # Get a random crop
        crop = self.crop_transform(original_image.convert("RGB"))

        # Resize for detector
        detector_img = crop.resize(self.detector_input_size, Image.Resampling.BICUBIC)

        # Resize for classifier
        classifier_img = crop.resize(
            self.classifier_input_size, Image.Resampling.BICUBIC
        )

        return {"detector_image": detector_img, "classifier_image": classifier_img}


class MultiScaleFeatureExtractor:
    """
    Helper class to extract multi-scale features from detector and classifier.
    """

    def __init__(self, detector, classifier, device="cuda"):
        self.detector = detector
        self.classifier = classifier
        self.device = device

        # We'll extract from these detector layers
        self.detector_layers = {
            "layer16": 16,  # P3/8
            "layer19": 19,  # P4/16
            "layer22": 22,  # P5/32
        }

    def extract_multi_scale_features(
        self, detector_images, classifier_images, classifier_bottleneck="model.8"
    ):
        """Extract features from multiple scales and classifier."""

        # Storage for features from each detector layer
        detector_features = {
            "layer16": None,  # P3/8
            "layer19": None,  # P4/16
            "layer22": None,  # P5/32
        }

        def create_hook(layer_name):
            def hook(module, input, output):
                detector_features[layer_name] = output.clone()

            return hook

        # Register hooks for all three layers
        hooks = [
            self.detector.model.model[16].register_forward_hook(create_hook("layer16")),
            self.detector.model.model[19].register_forward_hook(create_hook("layer19")),
            self.detector.model.model[22].register_forward_hook(create_hook("layer22")),
        ]

        try:
            # Run detector to capture features
            with torch.no_grad():
                _ = self.detector.run_examples_cuda(
                    detector_images, "model.22"
                )  # Just to trigger forward pass

                # Get classifier features
                classifier_features_raw = self.classifier.run_examples_cuda(
                    classifier_images, classifier_bottleneck
                )
                # Global average pooling for classifier features
                classifier_features = torch.mean(classifier_features_raw, dim=[2, 3])

        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

        return detector_features, classifier_features


class MultiScaleAlignmentLoss(nn.Module):
    """
    Loss function for training multi-scale alignment networks.
    Encourages aligned features to be similar across scales.
    """

    def __init__(self, temperature: float = 0.1, similarity_weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.similarity_weight = similarity_weight

    def forward(
        self,
        detector_features: torch.Tensor,
        classifier_features: torch.Tensor,
        raw_classifier_features: torch.Tensor,
        scales: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute alignment loss.

        Args:
            detector_features: Projected detector features [B, D]
            classifier_features: Classifier features [B, D]
            scales: Scale information for each sample

        Returns:
            Alignment loss scalar
        """
        # Compute similarity matrix
        similarity = (
            torch.mm(detector_features, classifier_features.T) / self.temperature
        )

        # Create labels (diagonal should be high)
        batch_size = detector_features.size(0)
        labels = torch.arange(batch_size, device=detector_features.device)

        # Cross-entropy loss for both directions
        loss_d2c = torch.nn.functional.cross_entropy(similarity, labels)
        loss_c2d = torch.nn.functional.cross_entropy(similarity.T, labels)

        alignment_loss = (loss_d2c + loss_c2d) / 2

        # Similarity preservation loss
        # Compute similarities in raw space
        raw_norm = torch.nn.functional.normalize(raw_classifier_features, dim=-1)
        raw_similarity = torch.mm(raw_norm, raw_norm.T)

        # Compute similarities in projected space
        projected_similarity = torch.mm(classifier_features, classifier_features.T)

        # MSE loss between similarity matrices
        similarity_loss = torch.nn.functional.mse_loss(
            projected_similarity, raw_similarity
        )

        # Combined loss
        total_loss = alignment_loss + self.similarity_weight * similarity_loss

        return total_loss, alignment_loss, similarity_loss


def train_multi_scale_alignment(
    detector_model_path: str,
    classifier_model_path: str,
    dataset,
    val_dataset,
    detector_bottlenecks: Optional[List[str]] = None,
    classifier_bottleneck: str = "model.8",
    projection_size: int = 128,
    batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    similarity_weight: float = 0.6,
    device: str = "cuda",
    output_subdir: str = "",
) -> MultiScaleAlignmentNetwork:
    """
    Train the multi-scale alignment network.

    Args:
        detector_model_path: Path to YOLO detector model
        classifier_model_path: Path to YOLO classifier model
        dataset: Training dataset
        val_dataset: Validation dataset
        detector_bottlenecks: List of detector layer names to use
        classifier_bottleneck: Classifier layer to extract features from
        projection_size: Size of the aligned feature space
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on

    Returns:
        Trained alignment network
    """
    if detector_bottlenecks is None:
        detector_bottlenecks = ["model.16", "model.19", "model.22"]

    if output_subdir == "":
        output_subdir = str(projection_size)

    # 1. Load models
    detector = UltralyticsModelWithGradients(detector_model_path)
    detector.to(device)
    detector.eval()

    classifier = UltralyticsModelWithGradients(classifier_model_path)
    classifier.to(device)
    classifier.eval()

    # 2. Get feature dimensions automatically
    import numpy as np

    # Use actual input sizes for each model
    detector_dummy_np = np.random.rand(1, 640, 640, 3).astype(np.float32)
    classifier_dummy_np = np.random.rand(1, 224, 224, 3).astype(np.float32)

    # Get classifier feature dimensions
    classifier_features_dummy = classifier.run_examples(
        classifier_dummy_np, classifier_bottleneck
    )
    classifier_channels = classifier_features_dummy.shape[1]

    # Get detector feature dimensions for each layer
    detector_channels = {}
    for bottleneck in detector_bottlenecks:
        detector_features_dummy = detector.run_examples(detector_dummy_np, bottleneck)
        layer_num = bottleneck.split(".")[-1]
        detector_channels[f"layer{layer_num}"] = detector_features_dummy.shape[1]

    print("Detector feature channels:")
    for layer, channels in detector_channels.items():
        print(f"  {layer}: {channels}")
    print(
        f"Classifier feature channels: {classifier_channels} (from {classifier_bottleneck})"
    )

    # 3. Initialize network with actual dimensions
    alignment_net = MultiScaleAlignmentNetwork(
        detector_channels_p3=detector_channels.get("layer16", 256),  # P3
        detector_channels_p4=detector_channels.get("layer19", 512),  # P4
        detector_channels_p5=detector_channels.get("layer22", 512),  # P5
        classifier_channels=classifier_channels,
        projection_size=projection_size,
    ).to(device)

    # 4. Initialize feature extractor
    feature_extractor = MultiScaleFeatureExtractor(detector, classifier, device)

    # 5. Prepare datasets and data loaders
    def collate_fn(batch):
        detector_images = []
        classifier_images = []
        for item in batch:
            detector_images.append(F.to_tensor(item["detector_image"]))
            classifier_images.append(F.to_tensor(item["classifier_image"]))

        return {
            "detector_image": torch.stack(detector_images),
            "classifier_image": torch.stack(classifier_images),
        }

    alignment_dataset = MultiScaleAlignmentDataset(
        dataset=dataset,
        detector_model_path=detector_model_path,
        classifier_model_path=classifier_model_path,
        detector_input_size=(640, 640),
        classifier_input_size=(224, 224),
        num_crops_per_image=3,
        device=device,
    )

    # Limit number of samples per epoch for manageable training
    max_samples_per_epoch = 5000

    def get_epoch_dataloader():
        import numpy as np
        from torch.utils.data import SubsetRandomSampler

        num_samples = min(len(alignment_dataset), max_samples_per_epoch)
        indices = np.random.choice(len(alignment_dataset), num_samples, replace=False)
        sampler = SubsetRandomSampler(indices)
        return DataLoader(
            alignment_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    # Validation dataset
    val_alignment_dataset = MultiScaleAlignmentDataset(
        dataset=val_dataset,
        detector_model_path=detector_model_path,
        classifier_model_path=classifier_model_path,
        detector_input_size=(640, 640),
        classifier_input_size=(224, 224),
        num_crops_per_image=2,
        device=device,
    )

    # Limit validation samples
    max_val_samples = min(len(val_alignment_dataset), max_samples_per_epoch // 4)
    val_indices = np.random.choice(
        len(val_alignment_dataset), max_val_samples, replace=False
    )
    val_sampler = SubsetRandomSampler(val_indices)
    val_dataloader = DataLoader(
        val_alignment_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 6. Loss and optimizer
    criterion = MultiScaleAlignmentLoss(
        temperature=0.1, similarity_weight=similarity_weight
    )
    optimizer = optim.Adam(alignment_net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 7. Training metrics tracking
    best_val_loss = float("inf")
    metrics_csv_path = Path("./alignment/multi_scale_training_metrics.csv")
    write_header = not metrics_csv_path.exists()

    # 8. Training loop
    print(f"Training multi-scale alignment for {num_epochs} epochs...")
    alignment_net.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        scale_counts = {"layer16": 0, "layer19": 0, "layer22": 0}

        dataloader = get_epoch_dataloader()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            # Get features from both models
            with torch.no_grad():
                detector_images = batch["detector_image"].to(device)
                classifier_images = batch["classifier_image"].to(device)

                # Extract multi-scale features
                detector_features, classifier_features = (
                    feature_extractor.extract_multi_scale_features(
                        detector_images, classifier_images, classifier_bottleneck
                    )
                )

                # Apply global average pooling to detector features
                detector_features_pooled = {}
                for scale, features in detector_features.items():
                    if features is not None:
                        detector_features_pooled[scale] = torch.mean(
                            features, dim=[2, 3]
                        )

                del detector_images, classifier_images

            # Process each scale separately
            batch_losses = []
            batch_alignment_losses = []
            batch_similarity_losses = []

            for scale in ["layer16", "layer19", "layer22"]:
                if (
                    scale not in detector_features_pooled
                    or detector_features_pooled[scale] is None
                ):
                    continue

                detector_feats = detector_features_pooled[scale]
                scale_counts[scale] += detector_feats.size(0)

                # Project features through appropriate scale network
                if scale == "layer16":  # P3
                    projected_detector = alignment_net.forward_p3(detector_feats)
                elif scale == "layer19":  # P4
                    projected_detector = alignment_net.forward_p4(detector_feats)
                else:  # layer22 -> P5
                    projected_detector = alignment_net.forward_p5(detector_feats)

                # Project classifier features
                projected_classifier = alignment_net.forward_classifier(
                    classifier_features
                )

                # Compute loss for this scale
                total_loss, alignment_loss, similarity_loss = criterion(
                    projected_detector,
                    projected_classifier,
                    classifier_features,  # Pass raw classifier features
                    [scale] * detector_feats.size(0),
                )

                batch_losses.append(total_loss)
                batch_alignment_losses.append(alignment_loss)
                batch_similarity_losses.append(similarity_loss)

            if batch_losses:
                # Average losses across scales - ensure we get a tensor
                total_batch_loss = torch.stack(batch_losses).mean()
                avg_alignment_loss = torch.stack(batch_alignment_losses).mean()
                avg_similarity_loss = torch.stack(batch_similarity_losses).mean()

                # Backward pass
                total_batch_loss.backward()
                optimizer.step()

                total_loss += total_batch_loss.item()

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "total": f"{total_batch_loss.item():.4f}",
                        "align": f"{avg_alignment_loss.item():.4f}",
                        "sim": f"{avg_similarity_loss.item():.4f}",
                        "p3": scale_counts["layer16"],
                        "p4": scale_counts["layer19"],
                        "p5": scale_counts["layer22"],
                    }
                )

        # End of epoch
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")

        # Validation loop
        alignment_net.eval()
        total_val_loss = 0.0
        val_scale_counts = {"layer16": 0, "layer19": 0, "layer22": 0}

        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f"Validation {epoch + 1}/{num_epochs}")
            for batch in val_pbar:
                # Get features from both models
                detector_images = batch["detector_image"].to(device)
                classifier_images = batch["classifier_image"].to(device)

                # Extract multi-scale features
                detector_features, classifier_features = (
                    feature_extractor.extract_multi_scale_features(
                        detector_images, classifier_images, classifier_bottleneck
                    )
                )

                # Apply global average pooling to detector features
                detector_features_pooled = {}
                for scale, features in detector_features.items():
                    if features is not None:
                        detector_features_pooled[scale] = torch.mean(
                            features, dim=[2, 3]
                        )

                # Process each scale separately
                batch_val_losses = []

                for scale in ["layer16", "layer19", "layer22"]:
                    if (
                        scale not in detector_features_pooled
                        or detector_features_pooled[scale] is None
                    ):
                        continue

                    detector_feats = detector_features_pooled[scale]
                    val_scale_counts[scale] += detector_feats.size(0)

                    # Project features through appropriate scale network
                    if scale == "layer16":  # P3
                        projected_detector = alignment_net.forward_p3(detector_feats)
                    elif scale == "layer19":  # P4
                        projected_detector = alignment_net.forward_p4(detector_feats)
                    else:  # layer22 -> P5
                        projected_detector = alignment_net.forward_p5(detector_feats)

                    # Project classifier features
                    projected_classifier = alignment_net.forward_classifier(
                        classifier_features
                    )

                    # Compute validation loss for this scale
                    total_val_loss, align_val_loss, sim_val_loss = criterion(
                        projected_detector,
                        projected_classifier,
                        classifier_features,  # Pass raw classifier features
                        [scale] * detector_feats.size(0),
                    )
                    batch_val_losses.append(total_val_loss)

                if batch_val_losses:
                    total_batch_val_loss = torch.stack(batch_val_losses).mean()
                    total_val_loss += total_batch_val_loss.item()
                    val_pbar.set_postfix(val_loss=f"{total_batch_val_loss.item():.4f}")

        avg_val_loss = (
            total_val_loss / len(val_dataloader)
            if len(val_dataloader) > 0
            else float("inf")
        )
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Average Validation Loss: {avg_val_loss:.4f}"
        )

        # Save the model with the best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = (
                Path("./alignment")
                / f"{output_subdir}/multi_scale_alignment_network_{os.path.splitext(os.path.basename(classifier_model_path))[0]}-{classifier_bottleneck}_{os.path.splitext(os.path.basename(detector_model_path))[0]}-multi_scale_best.pth"
            )
            save_path.parent.mkdir(exist_ok=True)
            torch.save(alignment_net.state_dict(), save_path)
            print(f"Saved best model to {save_path} (val_loss: {avg_val_loss:.4f})")

        # Export metrics to CSV
        with metrics_csv_path.open("a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["epoch", "train_loss", "val_loss"])
                write_header = False
            writer.writerow([epoch + 1, avg_loss, avg_val_loss])

        scheduler.step()
        alignment_net.train()  # Back to training mode

        print(
            f"  Scale distribution - P3: {scale_counts['layer16']}, P4: {scale_counts['layer19']}, P5: {scale_counts['layer22']}"
        )
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    # Save the final trained alignment network
    save_path = (
        Path("./alignment")
        / f"{output_subdir}/multi_scale_alignment_network_{os.path.splitext(os.path.basename(classifier_model_path))[0]}-{classifier_bottleneck}_{os.path.splitext(os.path.basename(detector_model_path))[0]}-multi_scale.pth"
    )
    save_path.parent.mkdir(exist_ok=True)
    torch.save(alignment_net.state_dict(), save_path)
    print(f"Final trained alignment network saved to {save_path}")

    # Save metadata for the best model
    metadata = {
        "detector_channels_p3": detector_channels.get("layer16", 256),
        "detector_channels_p4": detector_channels.get("layer19", 512),
        "detector_channels_p5": detector_channels.get("layer22", 512),
        "classifier_channels": classifier_channels,
        "projection_size": projection_size,
        "num_epochs": num_epochs,
        "best_val_loss": best_val_loss,
        "final_train_loss": avg_loss,
        "scale_distribution": scale_counts,
    }
    metadata_path = save_path.with_name(save_path.stem + "_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print("Multi-scale alignment training completed!")
    return alignment_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Scale Alignment Network")
    parser.add_argument(
        "--detector_model",
        type=str,
        default="models/yolo11s.pt",
        help="Path to YOLO detector model",
    )
    parser.add_argument(
        "--classifier_model",
        type=str,
        default="models/yolo11s-cls.pt",
        help="Path to YOLO classifier model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/imagenet-1k",
        help="Path to dataset",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1", help="Device to use for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument("--proj_size", type=int, default=512, help="Projection size")
    parser.add_argument(
        "--similarity_weight", type=float, default=0.6, help="Similarity weight"
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="",
        help="Output subdirectory for checkpoints",
    )

    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    from data.imagenet_1k.imagenet_1k import Imagenet1k

    dataset = Imagenet1k()
    dataset.download_and_prepare(base_path=f"{args.data_dir}/imagenet_1k", force_download=False)
    train_dataset = dataset.as_dataset(split="train")
    val_dataset = dataset.as_dataset(split="validation")

    # Train multi-scale alignment
    train_multi_scale_alignment(
        detector_model_path=args.detector_model,
        classifier_model_path=args.classifier_model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        projection_size=args.proj_size,
        similarity_weight=args.similarity_weight,
        output_subdir=args.output_subdir,
    )
