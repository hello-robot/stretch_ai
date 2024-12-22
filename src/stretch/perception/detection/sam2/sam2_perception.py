# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wget
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from stretch.core.abstract_perception import PerceptionModule
from stretch.perception.detection.utils import filter_depth, overlay_masks

PARENT_DIR = Path(__file__).resolve().parent
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = str(
    PARENT_DIR
    / "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = str(PARENT_DIR / "checkpoints" / "groundingdino_swint_ogc.pth")
MOBILE_SAM_CHECKPOINT_PATH = str(PARENT_DIR / "checkpoints" / "mobile_sam.pt")
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

_DEFAULT_MASK_GENERATOR_KWARGS = dict(
    points_per_side=32,
    pred_iou_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0].shape[0], sorted_anns[0].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


class SAM2Perception(PerceptionModule):
    def __init__(
        self,
        custom_vocabulary: List[str] = "['', 'dog', 'grass', 'sky']",
        gpu_device_id: Optional[int] = None,
        configuration: str = "l",
        checkpoint_file: str = MOBILE_SAM_CHECKPOINT_PATH,
        verbose=False,
        text_threshold: float = None,
        mask_generator_kwargs: Dict[str, Any] = _DEFAULT_MASK_GENERATOR_KWARGS,
    ):
        """Load trained Detic model for inference.

        Arguments:
            config_file: path to model config
            custom_vocabulary: if vocabulary="custom", this should be a comma-separated
             list of classes (as a single string)
            gpu_device_id: GPU ID to load the model on, None for default
            configuration: size of SAM2 model, default is t, also supporting s, b+, l
            checkpoint_file: path to model checkpoint
            verbose: whether to print out debug information
        """
        # TODO: set model arg in config
        # You can and are recommended to run `download_ckpts.sh` in "." folder
        # to download sam weights.
        # Here we still provide weight checking and automatic downloading.
        # By default, we use the largest model now.
        base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
        if configuration == "t":
            checkpoint = "./sam2_hiera_tiny.pt"
            model_cfg = "sam2_hiera_t.yaml"
            url = base_url + "sam2_hiera_tiny.pt"
        elif configuration == "s":
            checkpoint = "./sam2_hiera_small.pt"
            model_cfg = "sam2_hiera_s.yaml"
            url = base_url + "sam2_hiera_small.pt"
        elif configuration == "b+":
            checkpoint = "./sam2_hiera_base_plus.pt"
            model_cfg = "sam2_hiera_b+.yaml"
            url = base_url + "sam2_hiera_base_plus.pt"
        else:
            checkpoint = "./sam2_hiera_large.pt"
            model_cfg = "sam2_hiera_l.yaml"
            url = base_url + "sam2_hiera_large.pt"
        if not os.path.exists(checkpoint):
            wget.download(url, out=checkpoint)

        self.sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

        if not torch.cuda.is_available():
            if gpu_device_id is not None:
                print("Warning: CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
        elif gpu_device_id is None:
            device = torch.device("cuda")
        elif gpu_device_id < 0 or gpu_device_id >= torch.cuda.device_count():
            print(
                f"Warning: Invalid GPU device ID {gpu_device_id}. Falling back to default CUDA device."
            )
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{gpu_device_id}")

        self.sam2_predictor = build_sam2(
            model_cfg, checkpoint, device=device, apply_postprocessing=False
        )
        self._verbose = verbose

        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_predictor)

    def reset_vocab(self, new_vocab: List[str]):
        """Resets the vocabulary of Detic model allowing you to change detection on
        the fly. Note that previous vocabulary is not preserved.
        Args:
            new_vocab: list of strings representing the new vocabulary
            vocab_type: one of "custom" or "coco"; only "custom" supported right now
        """
        self.custom_vocabulary = new_vocab

    def generate(self, image, min_area=500):  # TODO: figure out how to set min_area in config
        masks = self.mask_generator.generate(image)
        returned_masks = []
        for mask in masks:
            # filter out masks that are too small
            if mask["area"] >= min_area:
                returned_masks.append(mask["segmentation"])
        return returned_masks

    # Prompting SAM with detected boxes
    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """
        Get masks for all detected bounding boxes using SAM
        Arguments:
            image: image of shape (H, W, 3)
            xyxy: bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        Returns:
            masks: masks of shape (N, H, W)
        """
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def predict(
        self,
        rgb=None,
        depth=None,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Get masks using SAM
        Arguments:
            image: image of shape (H, W, 3)
        Returns:
            masks: masks of shape (N, H, W)
        """
        if self._verbose:
            print("SAM2 is segmenting the image...")

        height, width, _ = rgb.shape
        image = rgb
        image = np.array(image)
        if not image.dtype == np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = image.astype(np.uint8)

        masks = np.array(self.generate(image))

        # from PIL import Image
        # for mask in masks:
        #     mm = np.expand_dims(mask, axis=-1)
        #     image_array = np.array(mm * image, dtype=np.uint8)
        #     image_debug = Image.fromarray(image_array)
        #     image_debug.show()

        # plt.figure(figsize=(20, 20))
        # plt.imshow(image)
        # plt.savefig("original.png", dpi=100)
        # show_anns(masks)
        # plt.axis("off")
        # plt.savefig("segmented.png", dpi=100)

        masks = sorted(masks, key=lambda x: np.count_nonzero(x), reverse=True)
        if depth_threshold is not None and depth is not None:
            masks = np.array([filter_depth(mask, depth, depth_threshold) for mask in masks])
        semantic_map, instance_map = overlay_masks(masks, np.zeros(len(masks)), (height, width))

        task_observations = dict()
        task_observations["instance_map"] = instance_map
        # random filling object classes -- right now using cups
        task_observations["instance_classes"] = np.full(len(masks), 31)
        task_observations["instance_scores"] = np.ones(len(masks))
        task_observations["semantic_frame"] = None
        return semantic_map.astype(int), instance_map.astype(int), task_observations

    def is_semantic(self):
        return True

    def is_instance(self):
        return True
