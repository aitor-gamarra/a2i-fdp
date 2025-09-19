import argparse
import os
import pickle
from pathlib import Path

import ace_helpers
from ace import ConceptDiscovery
from ace_helpers import get_class_name
from data.imagenet_1k.imagenet_1k import Imagenet1k
from models import ImageNetClass
from ultralytics_model_with_gradients import UltralyticsModelWithGradients

DEFAULT_MODEL_PATH = "models/yolo11s-cls.pt"
DEFAULT_IMAGENET_SPLIT = "train"

BOTTLENECKS = [
    "model.9",
    "model.7",
    "model.8",
    "model.6",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACE Concept Discovery")

    parser.add_argument(
        "--device",
        dest="device",
        default="cuda",
        help="Device to use (e.g., cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the model file",
    )
    parser.add_argument(
        "--class_idx",
        dest="class_idx",
        type=int,
        default=779,
        help="Index of the target class",
    )
    parser.add_argument(
        "--num_discovery_imgs",
        type=int,
        default=50,
        help="Number of images used for concept discovery (was 50)",
    )
    parser.add_argument(
        "--max_imgs",
        type=int,
        default=100,
        help="Maximum number of images per discovered concept (was 100)",
    )
    parser.add_argument(
        "--min_imgs",
        type=int,
        default=50,
        help="Minimum number of images per discovered concept (was 50)",
    )
    parser.add_argument(
        "--num_random_exp",
        type=int,
        default=60,
        help="Number of random concept repetitions for CAV statistical testing (was 60)",
    )
    parser.add_argument(
        "--no_channel_mean",
        action="store_true",
        help="Disable channel_mean reduction (keep full channel activations for clustering)",
    )
    parser.add_argument(
        "--imagenet_split",
        dest="imagenet_split",
        default=DEFAULT_IMAGENET_SPLIT,
        help="ImageNet split to use (train or validation)",
    )

    parser.add_argument(
        "--ace_data_dir",
        dest="ace_data_dir",
        default="ace_data_dir",
        help="Path to store the ACE data",
    )

    args = parser.parse_args()

    cls_model = UltralyticsModelWithGradients(args.model_path)
    cls_model.eval()

    target_class = ImageNetClass(
        get_class_name("dataset_data/ImageNet.yaml", args.class_idx), args.class_idx
    )

    data_dir = Path(args.ace_data_dir)

    print(
        f"Loading ImageNet images for class idx={target_class.index} ({target_class.name})..."
    )
    dataset = Imagenet1k()
    dataset.download_and_prepare(base_path=data_dir / "imagenet_1k", force_download=False)
    dataset = dataset.as_dataset(split=args.imagenet_split)

    cd = ConceptDiscovery(
        model=cls_model,
        target_class=target_class,
        random_concept="random_discovery",
        bottlenecks=BOTTLENECKS,
        dataset=dataset,
        data_dir=data_dir,
        device=args.device,
        num_random_exp=args.num_random_exp,
        channel_mean=not args.no_channel_mean,
        max_imgs=args.max_imgs,
        min_imgs=args.min_imgs,
        num_discovery_imgs=args.num_discovery_imgs,
        num_workers=15,
        average_image_value=255 / 2.0,
    )

    print("Creating patches...")
    cd.create_patches()

    print("Discovering concepts...")
    cd.discover_concepts(method="KM")

    del cd.image_numbers
    del cd.patches

    print("Saving concepts...")
    ace_helpers.save_concepts(cd, data_dir / "concepts")

    print("Calculating CAVs scores...")
    cav_accuracies = cd.cavs(min_acc=0.0)

    print("Calculating TCAV scores...")
    scores = cd.tcavs(test=False)

    # Plot examples of discovered concepts
    print("Plotting examples...")
    results_dir = data_dir / "results_summaries"
    os.makedirs(results_dir, exist_ok=True)

    for bn in cd.bottlenecks:
        if len(cd.dic[bn]["concepts"]) > 0:
            ace_helpers.plot_concepts(cd, bn, 10, address=results_dir, scores=scores)

    # Delete concepts that don't pass statistical testing
    print("Removing non-relevant concepts...")
    cd.test_and_remove_concepts(scores)

    # Plot examples of filtered discovered concepts
    print("Plotting (filtered) examples...")
    filtered_results_dir = results_dir / "filtered"
    os.makedirs(filtered_results_dir, exist_ok=True)

    for bn in cd.bottlenecks:
        if len(cd.dic[bn]["concepts"]) > 0:
            ace_helpers.plot_concepts(
                cd, bn, 10, address=filtered_results_dir, scores=scores
            )

    with open(data_dir / f"concept_discovery_{target_class.name}.pkl", "wb") as f:
        cd.model = None
        pickle.dump(cd, f)

    print(f"ACE analysis complete. Results saved to {data_dir}")
    print("Total concepts remaining after filtering:")
    for bn in cd.bottlenecks:
        print(f"  {bn}: {len(cd.dic[bn]['concepts'])} concepts")
