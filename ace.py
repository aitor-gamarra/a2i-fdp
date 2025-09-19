import os
from pathlib import Path

import numpy as np
import scipy.stats as stats
import skimage.segmentation as segmentation
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as metrics
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from numpy.typing import NDArray
from PIL import Image
from tcav import cav
from tqdm import tqdm

from ace_helpers import (
    get_acts_from_image_dataset,
    get_acts_from_images,
    pil_to_skimage,
)
from models import ImageNetClass
from ultralytics_model_with_gradients import UltralyticsModelWithGradients


class ConceptDiscovery(object):
    def __init__(
        self,
        model: UltralyticsModelWithGradients,
        target_class: ImageNetClass,
        random_concept: str,
        bottlenecks: list[str],
        dataset: Dataset | DatasetDict,
        data_dir: Path,
        device: str = "cuda",
        num_random_exp: int = 2,
        channel_mean: bool = True,
        max_imgs: int = 40,
        min_imgs: int = 20,
        num_discovery_imgs: int = 40,
        num_workers: int = 15,
        average_image_value: float = 255 / 2.0,
    ):
        """Runs concept discovery for a given class in a trained model.

        For a trained classification model, the ConceptDiscovery class first
        performs unsupervised concept discovery using examples of one of the classes
        in the network.

        Args:
            model: A trained classification model on which we run the concept
                            discovery algorithm
            target_class: Name of the one of the classes of the network
            random_concept: A concept made of random images (used for statistical
                                            test) e.g. "random500_199"
            bottlenecks: a list of bottleneck layers of the model for which the cocept
                                        discovery stage is performed
            sess: Model's tensorflow session
            source_dir: This directory that contains folders with images of network's
                                    classes.
            activation_dir: directory to save computed activations
            cav_dir: directory to save CAVs of discovered and random concepts
            num_random_exp: Number of random counterparts used for calculating several
                                            CAVs and TCAVs for each concept (to make statistical
                                                testing possible.)
            channel_mean: If true, for the unsupervised concept discovery the
                                        bottleneck activations are averaged over channels instead
                                        of using the whole acivation vector (reducing
                                        dimensionality)
            max_imgs: maximum number of images in a discovered concept
            min_imgs : minimum number of images in a discovered concept for the
                                    concept to be accepted
            num_discovery_imgs: Number of images used for concept discovery. If None,
                                                    will use max_imgs instead.
            num_workers: if greater than zero, runs methods in parallel with
                num_workers parallel threads. If 0, no method is run in parallel
                threads.
            average_image_value: The average value used for mean subtraction in the
                                                        nework's preprocessing stage.
        """
        self.model = model
        self.target_class = target_class
        self.bottlenecks = bottlenecks

        self.channel_mean = channel_mean
        self.random_concept = random_concept
        self.num_random_exp = num_random_exp
        self.max_imgs = max_imgs
        self.min_imgs = min_imgs
        self.num_discovery_imgs = num_discovery_imgs
        self.average_image_value = average_image_value
        self.num_workers = num_workers

        # Paths
        self.data_dir = data_dir
        self.cav_dir = data_dir / "cavs"
        self.activation_dir = data_dir / "acts"
        os.makedirs(self.cav_dir, exist_ok=True)
        os.makedirs(self.activation_dir, exist_ok=True)

        # TODO: Get this automatically
        self.image_shape = (224, 224)

        # Filter dataset for target class
        self.rnd_dataset: DatasetDict = dataset.shuffle(seed=42)
        self.dataset: DatasetDict = dataset.filter(
            lambda x: x["label"] == self.target_class.index, num_proc=self.num_workers
        )

        self.model.to(device)
        self.device = next(self.model.parameters()).device

    def create_patches(self, method="slic", param_dict=None):
        """Creates a set of image patches using superpixel methods.

        This method takes in the concept discovery images and transforms it to a
        dataset made of the patches of those images.

        Args:
            method: The superpixel method used for creating image patches. One of
                'slic', 'watershed', 'quickshift', 'felzenszwalb'.

            param_dict: Contains parameters of the superpixel method used in the form
                                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                                method.
        """
        if param_dict is None:
            param_dict = {}

        patches_dataset: list = []
        image_numbers: list = []
        patches: list = []

        # Get some images from dataset as discovery images
        self.discovery_images: Dataset = self.dataset.take(self.num_discovery_imgs)

        for fn, img_data in tqdm(
            enumerate(self.discovery_images),
            "Getting superpixels for images",
            total=self.num_discovery_imgs,
        ):
            image_superpixels, image_patches = self._return_superpixels(
                pil_to_skimage(img_data["image"]), method, param_dict
            )
            for superpixel, patch in zip(image_superpixels, image_patches):
                patches_dataset.append(superpixel)
                patches.append(patch)
                image_numbers.append(fn)
        self.patches_dataset, self.image_numbers, self.patches = (
            np.array(patches_dataset),
            np.array(image_numbers),
            np.array(patches),
        )

    def _return_superpixels(self, img, method="slic", param_dict=None):
        """Returns all patches for one image.

        Given an image, calculates superpixels for each of the parameter lists in
        param_dict and returns a set of unique superpixels by
        removing duplicates. If two patches have Jaccard similarity more than 0.5,
        they are concidered duplicates.

        Args:
            img: The input image
            method: superpixel method, one of slic, watershed, quichsift, or
                felzenszwalb
            param_dict: Contains parameters of the superpixel method used in the form
                                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                                method.
        Raises:
            ValueError: if the segementation method is invaled.
        """
        if param_dict is None:
            param_dict = {}
        if method == "slic":
            n_segmentss = param_dict.pop("n_segments", [15, 50, 80])
            n_params = len(n_segmentss)
            compactnesses = param_dict.pop("compactness", [20] * n_params)
            sigmas = param_dict.pop("sigma", [1.0] * n_params)
        elif method == "watershed":
            markerss = param_dict.pop("marker", [15, 50, 80])
            n_params = len(markerss)
            compactnesses = param_dict.pop("compactness", [0.0] * n_params)
        elif method == "quickshift":
            max_dists = param_dict.pop("max_dist", [20, 15, 10])
            n_params = len(max_dists)
            ratios = param_dict.pop("ratio", [1.0] * n_params)
            kernel_sizes = param_dict.pop("kernel_size", [10] * n_params)
        elif method == "felzenszwalb":
            scales = param_dict.pop("scale", [1200, 500, 250])
            n_params = len(scales)
            sigmas = param_dict.pop("sigma", [0.8] * n_params)
            min_sizes = param_dict.pop("min_size", [20] * n_params)
        else:
            raise ValueError("Invalid superpixel method!")

        unique_masks: list = []

        for i in range(n_params):
            param_masks = []
            if method == "slic":
                segments = segmentation.slic(
                    img,
                    n_segments=n_segmentss[i],
                    compactness=compactnesses[i],
                    sigma=sigmas[i],
                )
            elif method == "watershed":
                segments = segmentation.watershed(
                    img, markers=markerss[i], compactness=compactnesses[i]
                )
            elif method == "quickshift":
                segments = segmentation.quickshift(
                    img,
                    kernel_size=kernel_sizes[i],
                    max_dist=max_dists[i],
                    ratio=ratios[i],
                )
            elif method == "felzenszwalb":
                segments = segmentation.felzenszwalb(
                    img, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i]
                )
            for s in range(segments.max()):
                mask = (segments == s).astype(float)
                if np.mean(mask) > 0.001:
                    unique = True
                    for seen_mask in unique_masks:
                        jaccard = np.sum(seen_mask * mask) / np.sum(
                            (seen_mask + mask) > 0
                        )
                        if jaccard > 0.5:
                            unique = False
                            break
                    if unique:
                        param_masks.append(mask)
            unique_masks.extend(param_masks)

        superpixels: list[NDArray] = []
        patches: list[NDArray] = []

        while unique_masks:
            superpixel, patch = self._extract_patch(img, unique_masks.pop())
            superpixels.append(superpixel)
            patches.append(patch)
        return superpixels, patches

    def _extract_patch(self, image, mask):
        """Extracts a patch out of an image.

        Args:
            image: The original image
            mask: The binary mask of the patch area

        Returns:
            image_resized: The resized patch such that its boundaries touches the
                image boundaries
            patch: The original patch. Rest of the image is padded with average value
        """
        mask_expanded = np.expand_dims(mask, -1)
        patch = (
            mask_expanded * image
            + (1 - mask_expanded) * float(self.average_image_value) / 255
        )
        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
        image_resized = (
            np.array(image.resize(self.image_shape, Image.Resampling.BICUBIC)).astype(
                float
            )
            / 255
        )
        return image_resized, patch

    def _patch_activations(self, imgs, bottleneck, bs=100, channel_mean=None):
        """Returns activations of a list of imgs.

        Args:
            imgs: List/array of images to calculate the activations of
            bottleneck: Name of the bottleneck layer of the model where activations
                are calculated
            bs: The batch size for calculating activations. (To control computational
                cost)
            channel_mean: If true, the activations are averaged across channel.

        Returns:
            The array of activations
        """
        if channel_mean is None:
            channel_mean = self.channel_mean

        outputs: list = []
        for i in range(int(imgs.shape[0] / bs) + 1):
            outputs.append(
                self.model.run_examples(imgs[i * bs : (i + 1) * bs], bottleneck)
            )
        output = torch.cat(outputs, 0)
        if channel_mean and len(output.shape) > 3:
            # TODO: Check if these dimensions work for us
            output = torch.mean(output, (2, 3))
        else:
            output = output.reshape([output.shape[0], -1])
        return output

    def _cluster(self, acts, method="KM", param_dict=None):
        """Runs unsupervised clustering algorithm on concept actiavtations.

        Args:
            acts: activation vectors of datapoints points in the bottleneck layer.
                E.g. (number of clusters,) for Kmeans
            method: clustering method. We have:
                'KM': Kmeans Clustering
                'AP': Affinity Propagation
                'SC': Spectral Clustering
                'MS': Mean Shift clustering
                'DB': DBSCAN clustering method
            param_dict: Contains superpixl method's parameters. If an empty dict is
                                 given, default parameters are used.

        Returns:
            asg: The cluster assignment label of each data points
            cost: The clustering cost of each data point
            centers: The cluster centers. For methods like Affinity Propagetion
            where they do not return a cluster center or a clustering cost, it
            calculates the medoid as the center    and returns distance to center as
            each data points clustering cost.

        Raises:
            ValueError: if the clustering method is invalid.
        """
        if param_dict is None:
            param_dict = {}

        centers = None
        acts_np = acts.cpu().numpy() if isinstance(acts, torch.Tensor) else acts

        if method == "KM":
            n_clusters = param_dict.pop("n_clusters", 25)
            km = cluster.KMeans(n_clusters)
            d = km.fit(acts_np)
            centers = km.cluster_centers_
            d = np.linalg.norm(
                np.expand_dims(acts_np, 1) - np.expand_dims(centers, 0), ord=2, axis=-1
            )
            asg, cost = np.argmin(d, -1), np.min(d, -1)
        elif method == "AP":
            damping = param_dict.pop("damping", 0.5)
            ca = cluster.AffinityPropagation(damping=damping)
            ca.fit(acts_np)
            centers = ca.cluster_centers_
            d = np.linalg.norm(
                np.expand_dims(acts_np, 1) - np.expand_dims(centers, 0), ord=2, axis=-1
            )
            asg, cost = np.argmin(d, -1), np.min(d, -1)
        elif method == "MS":
            ms = cluster.MeanShift(n_jobs=self.num_workers)
            asg = ms.fit_predict(acts_np)
        elif method == "SC":
            n_clusters = param_dict.pop("n_clusters", 25)
            sc = cluster.SpectralClustering(
                n_clusters=n_clusters, n_jobs=self.num_workers
            )
            asg = sc.fit_predict(acts_np)
        elif method == "DB":
            eps = param_dict.pop("eps", 0.5)
            min_samples = param_dict.pop("min_samples", 20)
            sc = cluster.DBSCAN(eps, min_samples=min_samples, n_jobs=self.num_workers)
            asg = sc.fit_predict(acts_np)
        else:
            raise ValueError("Invalid Clustering Method!")

        if centers is None:  ## If clustering returned cluster centers, use medoids
            centers = np.zeros((asg.max() + 1, acts.shape[1]))
            cost = np.zeros(len(acts))
            for cluster_label in range(asg.max() + 1):
                cluster_idxs = np.where(asg == cluster_label)[0]
                cluster_points = acts[cluster_idxs]
                pw_distances = metrics.euclidean_distances(cluster_points)
                centers[cluster_label] = cluster_points[
                    np.argmin(np.sum(pw_distances, -1))
                ]
                cost[cluster_idxs] = np.linalg.norm(
                    acts[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
                    ord=2,
                    axis=-1,
                )
        return asg, cost, centers

    def discover_concepts(self, method="KM", activations=None, param_dicts=None):
        """Discovers the frequent occurring concepts in the target class.

            Calculates self.dic, a dicationary containing all the informations of the
            discovered concepts in the form of {'bottleneck layer name: bn_dic} where
            bn_dic itself is in the form of {'concepts:list of concepts,
            'concept name': concept_dic} where the concept_dic is in the form of
            {'images': resized patches of concept, 'patches': original patches of the
            concepts, 'image_numbers': image id of each patch}

        Args:
            method: Clustering method.
            activations: If activations are already calculated. If not calculates
                                     them. Must be a dictionary in the form of {'bn':array, ...}
            param_dicts: A dictionary in the format of {'bottleneck':param_dict,...}
                                     where param_dict contains the clustering method's parametrs
                                     in the form of {'param1':value, ...}. For instance for Kmeans
                                     {'n_clusters':25}. param_dicts can also be in the format
                                     of param_dict where same parameters are used for all
                                     bottlenecks.
        """
        if param_dicts is None:
            param_dicts = {}

        if set(param_dicts.keys()) != set(self.bottlenecks):
            param_dicts = {bn: param_dicts for bn in self.bottlenecks}

        self.dic = {}  ## The main dictionary of the ConceptDiscovery class.

        for bn in self.bottlenecks:
            print(f"Processing bottleneck: {bn}")
            bn_dic = {}

            if activations is None or bn not in activations.keys():
                bn_activations = self._patch_activations(self.patches_dataset, bn)
            else:
                bn_activations = activations[bn]

            bn_dic["label"], bn_dic["cost"], centers = self._cluster(
                bn_activations, method, param_dicts[bn]
            )

            concept_number, bn_dic["concepts"] = 0, []

            for i in tqdm(range(bn_dic["label"].max() + 1)):
                label_idxs = np.where(bn_dic["label"] == i)[0]
                if len(label_idxs) > self.min_imgs:
                    concept_costs = bn_dic["cost"][label_idxs]
                    concept_idxs = label_idxs[
                        np.argsort(concept_costs)[: self.max_imgs]
                    ]
                    concept_image_numbers = set(self.image_numbers[label_idxs])
                    discovery_size = len(self.discovery_images)
                    highly_common_concept = len(concept_image_numbers) > 0.5 * len(
                        label_idxs
                    )
                    mildly_common_concept = len(concept_image_numbers) > 0.25 * len(
                        label_idxs
                    )
                    mildly_populated_concept = (
                        len(concept_image_numbers) > 0.25 * discovery_size
                    )
                    cond2 = mildly_populated_concept and mildly_common_concept
                    non_common_concept = len(concept_image_numbers) > 0.1 * len(
                        label_idxs
                    )
                    highly_populated_concept = (
                        len(concept_image_numbers) > 0.5 * discovery_size
                    )
                    cond3 = non_common_concept and highly_populated_concept
                    if highly_common_concept or cond2 or cond3:
                        concept_number += 1
                        concept = "{}_concept{}".format(
                            self.target_class.name, concept_number
                        )
                        bn_dic["concepts"].append(concept)
                        bn_dic[concept] = {
                            "images": self.patches_dataset[concept_idxs],
                            "patches": self.patches[concept_idxs],
                            "image_numbers": self.image_numbers[concept_idxs],
                        }
                        bn_dic[concept + "_center"] = centers[i]
            bn_dic.pop("label", None)
            bn_dic.pop("cost", None)
            self.dic[bn] = bn_dic

    def _random_concept_activations(self, bottleneck, random_concept):
        """Wrapper for computing or loading activations of random concepts.

        Takes care of making, caching (if desired) and loading activations.

        Args:
            bottleneck: The bottleneck layer name
            random_concept: Name of the random concept e.g. "random500_0"

        Returns:
            A nested dict in the form of {concept:{bottleneck:activation}}
        """
        rnd_acts_path = os.path.join(
            self.activation_dir, f"acts_{random_concept}_{bottleneck}"
        )

        if not os.path.exists(rnd_acts_path):
            rnd_imgs = self.rnd_dataset.take(self.max_imgs)

            acts = get_acts_from_image_dataset(rnd_imgs, self.model, bottleneck)

            if isinstance(acts, torch.Tensor):
                acts = acts.cpu().numpy()

            with open(rnd_acts_path, "wb") as f:
                np.save(f, acts, allow_pickle=False)

            del acts
            del rnd_imgs

        return np.load(rnd_acts_path).squeeze()

    def _calculate_cav(self, c, r, bn, act_c, ow, directory=None):
        """Calculates a sinle cav for a concept and a one random counterpart.

        Args:
            c: conept name
            r: random concept name
            bn: the layer name
            act_c: activation matrix of the concept in the 'bn' layer
            ow: overwrite if CAV already exists
            directory: to save the generated CAV

        Returns:
            The accuracy of the CAV
        """
        if directory is None:
            directory = self.cav_dir

        act_c = act_c.cpu().numpy() if isinstance(act_c, torch.Tensor) else act_c
        act_r = self._random_concept_activations(bn, r)

        cav_instance = cav.get_or_train_cav(
            [c, r],
            bn,
            {c: {bn: act_c}, r: {bn: act_r}},
            cav_dir=directory,
            overwrite=ow,
        )
        return cav_instance.accuracies["overall"]

    def _concept_cavs(self, bn, concept, activations, randoms=None, ow=True):
        """Calculates CAVs of a concept versus all the random counterparts.

        Args:
            bn: bottleneck layer name
            concept: the concept name
            activations: activations of the concept in the bottleneck layer
            randoms: None if the class random concepts are going to be used
            ow: If true, overwrites the existing CAVs

        Returns:
            A dict of cav accuracies in the form of {'bottleneck layer':
            {'concept name':[list of accuracies], ...}, ...}
        """
        if randoms is None:
            randoms = ["random500_{}".format(i) for i in np.arange(self.num_random_exp)]

        accs = []
        for rnd in randoms:
            accs.append(self._calculate_cav(concept, rnd, bn, activations, ow))
        return accs

    def cavs(self, min_acc=0.0, ow=True):
        """Calculates cavs for all discovered concepts.

        This method calculates and saves CAVs for all the discovered concepts
        versus all random concepts in all the bottleneck layers

        Args:
            min_acc: Delete discovered concept if the average classification accuracy
                of the CAV is less than min_acc
            ow: If True, overwrites an already calcualted cav.

        Returns:
            A dicationary of classification accuracy of linear boundaries orthogonal
            to cav vectors
        """
        acc = {bn: {} for bn in self.bottlenecks}
        concepts_to_delete = []
        for bn in self.bottlenecks:
            for concept in self.dic[bn]["concepts"]:
                concept_imgs = self.dic[bn][concept]["images"]
                concept_acts = get_acts_from_images(concept_imgs, self.model, bn)
                acc[bn][concept] = self._concept_cavs(bn, concept, concept_acts, ow=ow)
                if np.mean(acc[bn][concept]) < min_acc:
                    concepts_to_delete.append((bn, concept))
            target_class_acts = get_acts_from_image_dataset(
                self.discovery_images, self.model, bn
            )
            acc[bn][self.target_class.name] = self._concept_cavs(
                bn, self.target_class.name, target_class_acts, ow=ow
            )
            rnd_acts = self._random_concept_activations(bn, self.random_concept)
            acc[bn][self.random_concept] = self._concept_cavs(
                bn, self.random_concept, rnd_acts, ow=ow
            )
        for bn, concept in concepts_to_delete:
            self.delete_concept(bn, concept)
        return acc

    def load_cav_direction(self, c, r, bn, directory=None):
        """Loads an already computed cav.

        Args:
            c: concept name
            r: random concept name
            bn: bottleneck layer
            directory: where CAV is saved

        Returns:
            The cav instance
        """
        if directory is None:
            directory = self.cav_dir

        model_type = "linear"
        alpha = 0.01

        cav_key = cav.CAV.cav_key([c, r], bn, model_type, alpha)
        cav_path = os.path.join(self.cav_dir, cav_key.replace("/", ".") + ".pkl")

        vector = cav.CAV.load_cav(cav_path).cavs[0]

        return np.expand_dims(vector, 0) / np.linalg.norm(vector, ord=2)

    def _sort_concepts(self, scores):
        for bn in self.bottlenecks:
            tcavs = []
            for concept in self.dic[bn]["concepts"]:
                tcavs.append(np.mean(scores[bn][concept]))
            concepts = []
            for idx in np.argsort(tcavs)[::-1]:
                concepts.append(self.dic[bn]["concepts"][idx])
            self.dic[bn]["concepts"] = concepts

    def _return_gradients(self, images_dataset):
        """For the given images calculates the gradient tensors.

        Args:
            images: Images for which we want to calculate gradients.

        Returns:
            A dictionary of images gradients in all bottleneck layers.
        """

        gradients = {}
        class_id = self.target_class.index

        new_imgs = []
        for item in images_dataset:
            img = item["image"]
            img_array = pil_to_skimage(img, shape=(224, 224))
            new_imgs.append(img_array)
        imgs_np = np.stack(new_imgs)

        for bn in self.bottlenecks:
            bn_grads = []
            for img in imgs_np:
                grads = self.model.get_gradient(img, class_id, bn).reshape(-1)
                bn_grads.append(grads)
            gradients[bn] = np.array(bn_grads)
        return gradients

    def _tcav_score(self, bn, concept, rnd, gradients):
        """Calculates and returns the TCAV score of a concept.

        Args:
            bn: bottleneck layer
            concept: concept name
            rnd: random counterpart
            gradients: Dict of gradients of tcav_score_images

        Returns:
            TCAV score of the concept with respect to the given random counterpart
        """
        vector = self.load_cav_direction(concept, rnd, bn)
        prod = np.sum(gradients[bn] * vector, -1)
        return np.mean(prod < 0)

    def tcavs(self, test=False, sort=True, tcav_score_images=None):
        """Calculates TCAV scores for all discovered concepts and sorts concepts.

        This method calculates TCAV scores of all the discovered concepts for
        the target class using all the calculated cavs. It later sorts concepts
        based on their TCAV scores.

        Args:
            test: If true, perform statistical testing and removes concepts that don't
                pass
            sort: If true, it will sort concepts in each bottleneck layers based on
                average TCAV score of the concept.
            tcav_score_images: Target class images used for calculating tcav scores.
                If None, the target class source directory images are used.

        Returns:
            A dictionary of the form {'bottleneck layer':{'concept name':
            [list of tcav scores], ...}, ...} containing TCAV scores.
        """

        tcav_scores = {bn: {} for bn in self.bottlenecks}
        randoms = ["random500_{}".format(i) for i in np.arange(self.num_random_exp)]

        if tcav_score_images is None:  # Load target class images if not given
            raw_imgs = self.dataset.select(range(2 * self.max_imgs))
            tcav_score_images = raw_imgs.select(range(self.max_imgs, 2 * self.max_imgs))

        gradients = self._return_gradients(tcav_score_images)

        for bn in self.bottlenecks:
            for concept in self.dic[bn]["concepts"] + [self.random_concept]:

                def t_func(rnd):
                    return self._tcav_score(bn, concept, rnd, gradients)

                tcav_scores[bn][concept] = [t_func(rnd) for rnd in randoms]

        if test:
            self.test_and_remove_concepts(tcav_scores)
        if sort:
            self._sort_concepts(tcav_scores)

        self.tcav_scores = tcav_scores
        return tcav_scores

    def do_statistical_testings(self, i_ups_concept, i_ups_random):
        """Conducts ttest to compare two set of samples.

        In particular, if the means of the two samples are staistically different.

        Args:
            i_ups_concept: samples of TCAV scores for concept vs. randoms
            i_ups_random: samples of TCAV scores for random vs. randoms

        Returns:
            p value
        """
        min_len = min(len(i_ups_concept), len(i_ups_random))
        _, p = stats.ttest_rel(i_ups_concept[:min_len], i_ups_random[:min_len])
        return p

    def test_and_remove_concepts(self, tcav_scores):
        """Performs statistical testing for all discovered concepts.

        Using TCAV socres of the discovered concepts versurs the random_counterpart
        concept, performs statistical testing and removes concepts that do not pass

        Args:
            tcav_scores: Calculated dicationary of tcav scores of all concepts
        """
        concepts_to_delete = []
        for bn in self.bottlenecks:
            for concept in self.dic[bn]["concepts"]:
                pvalue = self.do_statistical_testings(
                    tcav_scores[bn][concept], tcav_scores[bn][self.random_concept]
                )
                if pvalue > 0.01:
                    concepts_to_delete.append((bn, concept))
        for bn, concept in concepts_to_delete:
            self.delete_concept(bn, concept)

        self.tcav_scores = tcav_scores

    def delete_concept(self, bn, concept):
        """Removes a discovered concepts if it's not already removed.

        Args:
            bn: Bottleneck layer where the concepts is discovered.
            concept: concept name
        """
        self.dic[bn].pop(concept, None)
        if concept in self.dic[bn]["concepts"]:
            self.dic[bn]["concepts"].pop(self.dic[bn]["concepts"].index(concept))

    def _concept_profile(self, bn, activations, concept, randoms):
        """Transforms data points from activations space to concept space.

        Calculates concept profile of data points in the desired bottleneck
        layer's activation space for one of the concepts

        Args:
            bn: Bottleneck layer
            activations: activations of the data points in the bottleneck layer
            concept: concept name
            randoms: random concepts

        Returns:
            The projection of activations of all images on all CAV directions of
                the given concept
        """

        def t_func(rnd):
            products = (
                self.load_cav_direction(concept, rnd, bn) * activations.cpu().numpy()
            )
            return np.sum(products, -1)

        profiles = [t_func(rnd) for rnd in randoms]
        return np.stack(profiles, axis=-1)

    def find_profile(self, bn, images, mean=True):
        """Transforms images from pixel space to concept space.

        Args:
            bn: Bottleneck layer
            images: Data points to be transformed
            mean: If true, the profile of each concept would be the average inner
                product of all that concepts' CAV vectors rather than the stacked up
                version.

        Returns:
            The concept profile of input images in the bn layer.
        """
        profile = np.zeros(
            (len(images), len(self.dic[bn]["concepts"]), self.num_random_exp)
        )
        class_acts = get_acts_from_images(images, self.model, bn).reshape(
            [len(images), -1]
        )
        randoms = ["random500_{}".format(i) for i in range(self.num_random_exp)]
        for i, concept in enumerate(self.dic[bn]["concepts"]):
            profile[:, i, :] = self._concept_profile(bn, class_acts, concept, randoms)
        if mean:
            profile = np.mean(profile, -1)
        return profile
