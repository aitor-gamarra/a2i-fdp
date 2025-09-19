import os

import numpy as np
import yaml
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries
from sklearn.metrics.pairwise import cosine_similarity


def visualize_patch_grid(cd, bn, num_samples=5, out_dir="patch_grids"):
    """
    For each concept in the bottleneck, create a PNG with 5 patches and their mean/variance.
    Args:
        cd: ConceptDiscovery instance
        bn: bottleneck name (str)
        num_samples: number of patches per concept
        out_dir: directory to save PNGs
    """
    os.makedirs(out_dir, exist_ok=True)
    concepts = cd.dic[bn]["concepts"]
    for concept in concepts:
        patches = cd.dic[bn][concept]["patches"]
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
        for i in range(num_samples):
            if i < len(patches):
                patch = patches[i]
                mean = np.mean(patch)
                var = np.var(patch)
                axes[i].imshow(patch)
                axes[i].set_title(f"mean={mean:.2f}\nvar={var:.4f}", fontsize=10)
                axes[i].axis("off")
            else:
                axes[i].axis("off")
        fig.suptitle(f"{concept}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"patch_grid_{concept}.png"))
        plt.close(fig)


def pil_to_skimage(
    image: Image.Image, shape=(224, 224), resize=True, norm=True
) -> np.ndarray:
    if image.mode == "RGBA":
        image = image.convert("RGB")  # remove alpha
    elif image.mode == "L":
        image = image.convert("RGB")  # grayscale to RGB
    if resize:
        image = image.resize(shape, Image.BILINEAR)
    image = np.array(image)
    if norm:
        image = np.float32(image) / 255.0
    if not (len(image.shape) == 3 and image.shape[2] == 3):
        return None
    else:
        return image


def save_concepts(cd, concepts_dir):
    """Saves discovered concept's images or patches.

    Args:
      cd: The ConceptDiscovery instance the concepts of which we want to save
      concepts_dir: The directory to save the concept images
    """
    for bn in cd.bottlenecks:
        for concept in cd.dic[bn]["concepts"]:
            patches_dir = os.path.join(concepts_dir, bn + "_" + concept + "_patches")
            images_dir = os.path.join(concepts_dir, bn + "_" + concept)

            patches = (np.clip(cd.dic[bn][concept]["patches"], 0, 1) * 256).astype(
                np.uint8
            )
            images = (np.clip(cd.dic[bn][concept]["images"], 0, 1) * 256).astype(
                np.uint8
            )

            os.makedirs(patches_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)

            image_numbers = cd.dic[bn][concept]["image_numbers"]
            image_addresses, patch_addresses = [], []
            for i in range(len(images)):
                image_name = f"{i + 1:02d}_{image_numbers[i]}"
                patch_addresses.append(os.path.join(patches_dir, image_name + ".png"))
                image_addresses.append(os.path.join(images_dir, image_name + ".png"))

            save_images(patch_addresses, patches)
            save_images(image_addresses, images)


def save_images(addresses, images):
    """Save images in the addresses.

    Args:
      addresses: The list of addresses to save the images as or the address of the
        directory to save all images in. (list or str)
      images: The list of all images in numpy uint8 format.
    """
    if not isinstance(addresses, list):
        image_addresses = []
        for i, image in enumerate(images):
            image_name = f"{i + 1:03d}.png"
            image_addresses.append(os.path.join(addresses, image_name))
        addresses = image_addresses

    assert len(addresses) == len(images), "Invalid number of addresses"

    for address, image in zip(addresses, images):
        Image.fromarray(image).save(address, format="PNG")


def get_acts_from_image_dataset(
    imgs_dataset, model, bn_name, batch_size=32, device="cuda"
):
    new_imgs = []
    for item in imgs_dataset:
        img = item["image"]  # PIL image
        img_array = pil_to_skimage(img)
        new_imgs.append(img_array)
    imgs_np = np.stack(new_imgs)  # shape: [N, C, H, W]

    return get_acts_from_images(
        imgs_np,
        model,
        bn_name,
        batch_size,
        device,
    )


def get_acts_from_images(imgs, model, bn_name, batch_size=32, device="cuda"):
    """Run images through the model to get activations from a specific layer via hook.

    Args:
        imgs: A list or numpy array of images (assumed shape: [N, C, H, W])
        model: A PyTorch model
        hook: An activation hook object with a `.get()` method
        batch_size: Batch size for inference
        device: Device to run inference on ('cuda' or 'cpu')

    Returns:
        Torch tensor of activations from the specified bottleneck layer.
    """

    # TODO: Check if we need np.asarray() here
    return model.run_examples(imgs, bn_name).squeeze()


def find_bottleneck_idx(model, full_module_name):
    """
    Given full name like 'model.model.9.m.0.ffn.1.act',
    find the index i such that full_module_name starts with f'model.model.{i}'
    """
    prefix = "model.model."
    assert full_module_name.startswith(prefix)
    name_suffix = full_module_name[len(prefix) :]

    for idx, module in enumerate(model.model):
        base_name = f"{prefix}{idx}"
        if full_module_name.startswith(base_name):
            return idx
    return None


def save_ace_report(cd, accs, scores, address):
    """Saves TCAV scores.

    Saves the average CAV accuracies and average TCAV scores of the concepts
    discovered in ConceptDiscovery instance.

    Args:
      cd: The ConceptDiscovery instance.
      accs: The cav accuracy dictionary returned by cavs method of the
        ConceptDiscovery instance
      scores: The tcav score dictionary returned by tcavs method of the
        ConceptDiscovery instance
      address: The address to save the text file in.
    """
    report = "\n\n\t\t\t ---CAV accuracies---"
    for bn in cd.bottlenecks:
        report += "\n"
        for concept in cd.dic[bn]["concepts"]:
            report += "\n" + bn + ":" + concept + ":" + str(np.mean(accs[bn][concept]))
    with open(address, "w") as f:
        f.write(report)
    report = "\n\n\t\t\t ---TCAV scores---"
    for bn in cd.bottlenecks:
        report += "\n"
        for concept in cd.dic[bn]["concepts"]:
            # Check if enhanced statistical testing is available
            if hasattr(cd, "enhanced_statistical_testing"):
                test_results = cd.enhanced_statistical_testing(
                    scores[bn][concept], scores[bn][cd.random_concept]
                )
                pvalue = test_results["p_ttest"]  # Use t-test p-value for compatibility
            else:
                # Fallback to original method
                pvalue = cd.do_statistical_testings(
                    scores[bn][concept], scores[bn][cd.random_concept]
                )
            report += "\n{}:{}:{},{}".format(
                bn, concept, np.mean(scores[bn][concept]), pvalue
            )
    with open(address, "a") as f:
        f.write(report)


def plot_concepts(
    cd, bn, num=10, address=None, mode="diverse", concepts=None, scores=None
):
    """Plots examples of discovered concepts.

    Args:
      cd: The concept discovery instance
      bn: Bottleneck layer name
      num: Number of images to print out of each concept
      address: If not None, saves the output to the address as a .PNG image
      mode: If 'diverse', it prints one example of each of the target class images
        is coming from. If 'radnom', randomly samples exmples of the concept. If
        'max', prints out the most activating examples of that concept.
      concepts: If None, prints out examples of all discovered concepts.
        Otherwise, it should be either a list of concepts to print out examples of
        or just one concept's name
      scores: (optional) dictionary of TCAV scores, e.g. from cd.tcavs()

    Raises:
      ValueError: If the mode is invalid.
    """
    if concepts is None:
        concepts = cd.dic[bn]["concepts"]
    elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
        concepts = [concepts]
    num_concepts = len(concepts)
    plt.rcParams["figure.figsize"] = num * 2.1, 4.3 * num_concepts
    fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
    outer = gridspec.GridSpec(num_concepts, 1, wspace=0.0, hspace=0.3)
    for n, concept in enumerate(concepts):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, num, subplot_spec=outer[n], wspace=0, hspace=0.1
        )
        concept_images = cd.dic[bn][concept]["images"]
        concept_patches = cd.dic[bn][concept]["patches"]
        concept_image_numbers = cd.dic[bn][concept]["image_numbers"]
        if mode == "max":
            idxs = np.arange(len(concept_images))
        elif mode == "random":
            idxs = np.random.permutation(np.arange(len(concept_images)))
        elif mode == "diverse":
            idxs = []
            while True:
                seen = set()
                for idx in range(len(concept_images)):
                    if concept_image_numbers[idx] not in seen and idx not in idxs:
                        seen.add(concept_image_numbers[idx])
                        idxs.append(idx)
                if len(idxs) == len(concept_images):
                    break
        else:
            raise ValueError("Invalid mode!")
        idxs = idxs[:num]
        for i, idx in enumerate(idxs):
            ax = plt.Subplot(fig, inner[i])
            ax.imshow(concept_images[idx])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == int(num / 2):
                ax.set_title(concept)
                title = concept
                if scores is not None and bn in scores and concept in scores[bn]:
                    tcav_score = np.mean(scores[bn][concept])
                    title += f"\nTCAV: {tcav_score:.2f}"
                ax.set_title(title)
            ax.grid(False)
            fig.add_subplot(ax)
            ax = plt.Subplot(fig, inner[i + num])
            mask = 1 - (
                np.mean(concept_patches[idx] == float(cd.average_image_value) / 255, -1)
                == 1
            )

            image = cd.discovery_images["image"][int(concept_image_numbers[idx])]
            ax.imshow(
                mark_boundaries(
                    pil_to_skimage(image),
                    mask,
                    color=(1, 1, 0),
                    mode="thick",
                )
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(int(concept_image_numbers[idx])))
            ax.grid(False)
            fig.add_subplot(ax)
    plt.suptitle(bn)
    if address is not None:
        fig.savefig(address / f"{cd.target_class.name}_{bn}.png")
        plt.clf()
        plt.close(fig)


def apply_concepts_to_new_image(
    concept_discovery, img, bottleneck_layer, mean_profile=True
):
    """Apply discovered concepts to a new image and get concept activations."""
    img_array = img
    images = np.expand_dims(img_array, axis=0)

    concept_profile = concept_discovery.find_profile(
        bn=bottleneck_layer, images=images, mean=mean_profile
    )

    concept_names = concept_discovery.dic[bottleneck_layer]["concepts"]
    return concept_profile.squeeze(), concept_names


def get_concept_matches(
    concept_discovery,
    img,
    bottleneck_layer,
    method="slic",
    param_dict=None,
    top_concepts=3,
):
    """
    Comprehensive visualization showing:
    1. Original image with concept activations
    2. Top matching patches for each high-activation concept
    3. Concept examples from discovery

    Args:
        concept_discovery: Trained ConceptDiscovery instance
        img: The new image to analyze
        bottleneck_layer: Bottleneck layer to visualize
        method: Superpixel method for patch creation
        param_dict: Parameters for superpixel method
        top_concepts: Number of top-activated concepts to show
        figsize: Figure size
    """

    profile, concept_names = apply_concepts_to_new_image(
        concept_discovery, img, bottleneck_layer
    )

    # Find top activated concepts
    # top_indices = np.argsort(np.abs(profile))[::-1][:top_concepts]
    top_indices = np.argsort(profile)[::-1][:top_concepts]
    top_concept_names = [concept_names[i] for i in top_indices]
    top_activations = [profile[i] for i in top_indices]

    print("Top Concept Activations:")
    for name, activation in zip(top_concept_names, top_activations):
        print(f"{name}: {activation:.3f}")


def get_concept_matches_weighted(
    concept_discovery,
    img,
    bottleneck_layer,
    method="slic",
    param_dict=None,
    top_concepts=3,
):
    """
    Comprehensive visualization showing:
    1. Original image with concept activations
    2. Top matching patches for each high-activation concept
    3. Concept examples from discovery

    Args:
        concept_discovery: Trained ConceptDiscovery instance
        img: The new image to analyze
        bottleneck_layer: Bottleneck layer to visualize
        method: Superpixel method for patch creation
        param_dict: Parameters for superpixel method
        top_concepts: Number of top-activated concepts to show
        figsize: Figure size
    """

    tcav_scores = concept_discovery.tcav_scores[bottleneck_layer]

    profile, concept_names = apply_concepts_to_new_image(
        concept_discovery, img, bottleneck_layer
    )

    # Find top activated concepts
    top_indices = np.argsort(profile)[::-1][:top_concepts]
    top_concept_names = concept_names[:top_concepts]
    top_activations = profile[:top_concepts]

    print("Top Concept Activations:")
    for name, activation in zip(top_concept_names, top_activations):
        print(f"{name}: {(np.mean(tcav_scores[name]) * activation):.3f}")


def visualize_concept_matches_on_image(
    concept_discovery,
    img,
    bottleneck_layer,
    method="slic",
    param_dict=None,
    top_concepts=3,
    figsize=(20, 12),
):
    """
    Comprehensive visualization showing:
    1. Original image with concept activations
    2. Top matching patches for each high-activation concept
    3. Concept examples from discovery

    Args:
        concept_discovery: Trained ConceptDiscovery instance
        img: The new image to analyze
        bottleneck_layer: Bottleneck layer to visualize
        method: Superpixel method for patch creation
        param_dict: Parameters for superpixel method
        top_concepts: Number of top-activated concepts to show
        figsize: Figure size
    """

    # Get concept profile for the new image
    img_array = img

    profile, concept_names = apply_concepts_to_new_image(
        concept_discovery, img, bottleneck_layer
    )

    # Get patches from the new image
    patches, patch_activations, _ = find_concept_patches_in_new_image(
        concept_discovery, img, bottleneck_layer, method, param_dict
    )

    # Find top activated concepts
    # top_indices = np.argsort(np.abs(profile))[::-1][:top_concepts]
    top_indices = np.argsort(profile)[::-1][:top_concepts]
    top_concept_names = [concept_names[i] for i in top_indices]
    top_activations = [profile[i] for i in top_indices]

    # Create the visualization
    fig = plt.figure(figsize=figsize)

    # Main title
    fig.suptitle(f"Concept Analysis for New Image - {bottleneck_layer}", fontsize=16)

    # Layout: Original image + concept grid
    gs = fig.add_gridspec(
        2, top_concepts + 1, height_ratios=[1, 1], width_ratios=[2] + [1] * top_concepts
    )

    # Show original image
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(img_array)
    ax_orig.set_title("Original Image")
    ax_orig.axis("off")

    # Show concept activations as text
    activation_text = "Top Concept Activations:\n"
    for name, activation in zip(top_concept_names, top_activations):
        activation_text += f"{name}: {activation:.3f}\n"

    ax_orig.text(
        1.05,
        0.5,
        activation_text,
        transform=ax_orig.transAxes,
        verticalalignment="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
    )

    # Show patches from new image that match each top concept
    if len(patches) > 0:
        patch_matches = match_patches_to_concepts(
            concept_discovery, patch_activations, bottleneck_layer, top_k=1
        )

        for col, (concept_name, activation) in enumerate(
            zip(top_concept_names, top_activations)
        ):
            # Find patches that best match this concept
            matching_patches = []
            for i, matches in enumerate(patch_matches):
                if matches and matches[0][0] == concept_name:
                    matching_patches.append(
                        (i, matches[0][1])
                    )  # (patch_idx, similarity)

            # Sort by similarity and take top few
            matching_patches.sort(key=lambda x: x[1], reverse=True)

            # Show discovered concept examples (top row)
            ax_concept = fig.add_subplot(gs[0, col + 1])
            if concept_name in concept_discovery.dic[bottleneck_layer]:
                concept_patches = concept_discovery.dic[bottleneck_layer][concept_name][
                    "images"
                ]
                if len(concept_patches) > 0:
                    ax_concept.imshow(concept_patches[0])
            ax_concept.set_title(
                f"{concept_name}\nActivation: {activation:.3f}", fontsize=10
            )
            ax_concept.axis("off")

            # Show matching patch from new image (bottom row)
            ax_match = fig.add_subplot(gs[1, col + 1])
            if matching_patches:
                patch_idx = matching_patches[0][0]
                similarity = matching_patches[0][1]
                ax_match.imshow(patches[patch_idx])
                ax_match.set_title(
                    f"Best Match\nSimilarity: {similarity:.3f}", fontsize=10
                )
            else:
                ax_match.text(
                    0.5,
                    0.5,
                    "No\nMatching\nPatch",
                    ha="center",
                    va="center",
                    transform=ax_match.transAxes,
                )
                ax_match.set_title("No Match", fontsize=10)
            ax_match.axis("off")

    plt.tight_layout()
    plt.show()


def match_patches_to_concepts(
    concept_discovery, patch_activations, bottleneck_layer, top_k=5
):
    """
    Find which discovered concepts best match the patches from a new image.

    Args:
        concept_discovery: Trained ConceptDiscovery instance
        patch_activations: Activations of new image patches
        bottleneck_layer: Bottleneck layer being analyzed
        top_k: Number of top matching concepts to return per patch

    Returns:
        matches: List of top concept matches for each patch
    """

    if len(patch_activations) == 0:
        return []

    concepts = concept_discovery.dic[bottleneck_layer]["concepts"]
    matches = []

    for patch_act in patch_activations:
        concept_similarities = []

        for concept in concepts:
            # Get concept center (average activation)
            concept_center = concept_discovery.dic[bottleneck_layer][
                concept + "_center"
            ]

            # Calculate similarity (cosine similarity)
            similarity = cosine_similarity(
                patch_act.reshape(1, -1), concept_center.reshape(1, -1)
            )[0, 0]

            concept_similarities.append((concept, similarity))

        # Sort by similarity and take top k
        concept_similarities.sort(key=lambda x: x[1], reverse=True)
        matches.append(concept_similarities[:top_k])

    return matches


def find_concept_patches_in_new_image(
    concept_discovery, img, bottleneck_layer, method="slic", param_dict=None
):
    """
    Create patches from a new image using the same method as concept discovery.

    Args:
        concept_discovery: Trained ConceptDiscovery instance
        img: The new image
        bottleneck_layer: Bottleneck layer to analyze
        method: Superpixel method ('slic', 'watershed', 'quickshift', 'felzenszwalb')
        param_dict: Parameters for superpixel method

    Returns:
        patches: Array of image patches
        patch_activations: Activations of patches in the bottleneck layer
        original_image: The original image
    """

    # Load and resize the new image
    img_array = img

    # Create patches using the same method as discovery
    if param_dict is None:
        param_dict = {}

    # Use the internal method to create superpixels
    superpixels, patches = concept_discovery._return_superpixels(
        img_array, method, param_dict
    )

    # Get activations for these patches
    if len(patches) > 0:
        patches_array = np.array(superpixels)
        patch_activations = concept_discovery._patch_activations(
            patches_array, bottleneck_layer
        )
    else:
        patch_activations = np.array([])

    return np.array(superpixels), patch_activations, img_array


def get_class_name(yaml_path, index):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    # If names is a dict, use index as key; if it's a list, use index as position
    if isinstance(names, dict):
        return names.get(index, "Class not found")
    elif isinstance(names, list):
        return names[index] if 0 <= index < len(names) else "Class not found"
    else:
        return "Invalid format"


def get_coco_names(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return list(data.get("names", {}).values())
