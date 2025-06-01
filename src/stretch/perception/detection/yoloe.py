# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# (c) 2024 Hello Robot by Atharva Pusalkar
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOE

from stretch.core.abstract_perception import PerceptionModule
from stretch.core.interfaces import Observations
from stretch.perception.detection.scannet_200_classes import CLASS_LABELS_200
from stretch.perception.detection.utils import filter_depth, overlay_masks
from stretch.utils.image import Camera, camera_xyz_to_global_xyz


def merge_masks(masks, height, width) -> np.ndarray:
    """Merge masks into a single image.

    Arguments:
        masks: list of masks
        height: height of the image
        width: width of the image

    Returns:
        merged_mask: a single mask image
    """
    merged_mask = np.zeros((height, width), dtype=np.uint8)

    for i, mask in enumerate(masks, start=1):
        # Ensure mask has the correct shape
        if len(mask.shape) == 3:
            mask = mask[0]
        if mask.shape != (height, width):
            mask_resized = cv2.resize(mask, (width, height))
        else:
            mask_resized = mask
        # Add the mask to the merged image with a unique ID
        merged_mask[mask_resized] = i + 1

    return merged_mask


def draw_masks(masks, height, width):
    panoptic_masks = []
    for mask in masks:
        xy_coords = (mask * [width, height]).astype(np.int32)

        panotic_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw filled polygon
        cv2.fillPoly(panotic_mask, [xy_coords], color=1)

        # Add to list
        panoptic_masks.append(panotic_mask.astype(bool))

    return panoptic_masks


class YoloEPerception(PerceptionModule):
    def __init__(
        self,
        config_file=None,
        vocabulary="coco",
        class_list: Optional[Union[List[str], Tuple[str]]] = None,
        sem_gpu_id=0,
        verbose: bool = False,
        size: str = "s",
        confidence_threshold: Optional[float] = None,
    ):
        """Load trained YOLO model for inference.

        Arguments:
            config_file: path to model config
            vocabulary: currently one of "coco" for indoor coco categories or "custom"
             for a custom set of categories
            custom_vocabulary: if vocabulary="custom", this should be a comma-separated
             list of classes (as a single string)
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
        """
        self.verbose = verbose
        self.confidence = confidence_threshold if confidence_threshold is not None else 0.1

        if class_list is None:
            self.class_list = CLASS_LABELS_200

        checkpoint_file = f"yoloe-11{size}-seg.pt"
        self.model = YOLOE(checkpoint_file)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        if self.verbose:
            print(f"Loaded YOLO model from {checkpoint_file}")

        self.num_sem_categories = 80

    def reset_vocab(self, new_vocab: List[str], vocab_type="custom"):
        print("Resetting vocabulary not supported for YOLO")

    def predict(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
        confidence_threshold: Optional[float] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order - YOLO expects BGR)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """

        if isinstance(texts, str):
            texts = [texts]

        if texts is not None:
            self.model.set_classes(texts, self.model.get_text_pe(texts))
        else:
            self.model.set_classes(self.class_list, self.model.get_text_pe(self.class_list))

        if isinstance(rgb, np.ndarray):
            # This is expected
            pass
        elif isinstance(rgb, torch.Tensor):
            # Turn it into a numpy array
            rgb = rgb.numpy()
        else:
            raise ValueError(f"Expected rgb to be a numpy array or torch tensor, got {type(rgb)}")
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape
        if confidence_threshold is None:
            pred = self.model(image, verbose=self.verbose, conf=self.confidence)
        else:
            pred = self.model(image, verbose=self.verbose, conf=confidence_threshold)
        task_observations = dict()

        if pred[0].boxes is None:
            task_observations["semantic_frame"] = None
            return (
                np.zeros((rgb.shape[0], rgb.shape[1])),
                -1 * np.ones((rgb.shape[0], rgb.shape[1])),
                task_observations,
            )

        class_idcs = pred[0].boxes.cls.cpu().numpy()

        # Add some visualization code for YOLO
        if draw_instance_predictions:
            task_observations["semantic_frame"] = pred[0].plot(conf=False, labels=False, masks=True)
        else:
            task_observations["semantic_frame"] = None

        # Sort instances by mask size
        scores = pred[0].boxes.conf.cpu().numpy()

        masks = draw_masks(pred[0].masks.xyn, height, width)

        if depth_threshold is not None and depth is not None:
            masks = np.array([filter_depth(mask, depth, depth_threshold) for mask in masks])

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))

        semantic = semantic_map.astype(int)
        instance = instance_map.astype(int)
        task_observations["instance_map"] = instance_map
        task_observations["instance_classes"] = class_idcs
        task_observations["instance_scores"] = scores

        return semantic, instance, task_observations

    def is_semantic(self):
        return True

    def is_instance(self):
        return True

    def detect_object(
        self,
        rgb: Union[np.ndarray, torch.Tensor, Image.Image],
        text: Union[str, List[str]],
        confidence_threshold: Optional[float] = None,
        output_mask: Optional[bool] = True,
        visualize_mask: bool = False,
        mask_filename: Optional[str] = None,
        box_filename: Optional[str] = None,
    ):
        """Try to find target objects given one or many text queries.
        Arguments:
            rgb: ideally of shape (C, H, W), the pixel value should be integer between [0, 255]
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.numpy()
        if not isinstance(rgb, Image.Image):
            height, width, _ = rgb.shape
            image = Image.fromarray(rgb.astype(np.uint8))
        else:
            width, height = rgb.size
            image = rgb
        if not isinstance(text, list):
            text = [text]
        self.model.set_classes(text, self.model.get_text_pe(text))
        results = self.model.predict(image, conf=confidence_threshold)

        if output_mask:
            if results[0].masks is None or len(results[0].masks) == 0:
                return None, None
            else:
                masks = draw_masks(results[0].masks.xyn, height, width)
                return masks[0], results[0].boxes.xyxy.cpu().numpy()[0]
        else:
            return results[0].boxes.conf.cpu().numpy(), results[0].boxes.xyxy.cpu().numpy()

    def compute_obj_coord(
        self,
        text: str,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        camera_K: torch.Tensor,
        camera_pose: torch.Tensor,
        confidence_threshold: Optional[float] = None,
        depth_threshold: float = 3.0,
    ):
        height, width = depth.squeeze().shape
        camera = Camera.from_K(np.array(camera_K), width=width, height=height)
        camera_xyz = camera.depth_to_xyz(np.array(depth))
        xyz = torch.Tensor(camera_xyz_to_global_xyz(camera_xyz, np.array(camera_pose)))

        scores, boxes = self.detect_object(
            rgb=rgb, text=text, confidence_threshold=confidence_threshold, output_mask=False
        )
        for idx, (score, bbox) in enumerate(
            sorted(zip(scores, boxes), key=lambda x: x[0], reverse=True)
        ):

            tl_x, tl_y, br_x, br_y = bbox
            w, h = depth.shape
            tl_x, tl_y, br_x, br_y = (
                int(max(0, tl_x.item())),
                int(max(0, tl_y.item())),
                int(min(h, br_x.item())),
                int(min(w, br_y.item())),
            )

            if torch.min(depth[tl_y:br_y, tl_x:br_x].reshape(-1)) < depth_threshold:
                return torch.median(xyz[tl_y:br_y, tl_x:br_x].reshape(-1, 3), dim=0).values
        return None


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=["lvis", "openimages", "objects365", "coco", "custom"],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.45,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
