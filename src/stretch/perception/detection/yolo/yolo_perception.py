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
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from stretch.core.abstract_perception import PerceptionModule
from stretch.core.interfaces import Observations
from stretch.perception.detection.utils import filter_depth, overlay_masks
from stretch.utils.config import get_full_config_path


class YoloPerception(PerceptionModule):
    def __init__(
        self,
        config_file=None,
        vocabulary="coco",
        custom_vocabulary="",
        checkpoint_file=None,
        sem_gpu_id=0,
        verbose: bool = False,
        confidence_threshold: Optional[float] = None,
    ):
        """Load trained YOLO model for inference.

        Arguments:
            config_file: path to model config
            vocabulary: currently one of "coco" for indoor coco categories or "custom"
             for a custom set of categories
            custom_vocabulary: if vocabulary="custom", this should be a comma-separated
             list of classes (as a single string)
            checkpoint_file: path to model checkpoint
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
        """
        self.verbose = verbose

        if checkpoint_file is None:
            checkpoint_file = get_full_config_path("perception/yolo_world/yolov8s-seg.pt")

        # Check if checkpoint file exists
        if not Path(checkpoint_file).exists():
            # Make parent directory
            Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            # Download the model
            os.system(
                f"wget -O {checkpoint_file}"
                "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt"
            )

        self.yolo = YOLO(checkpoint_file, task="segment", verbose=verbose)

        if self.verbose:
            print(f"Loaded YOLO model from {checkpoint_file}")

        if vocabulary == "custom":
            # assert custom_vocabulary != ""
            # string_args += f""" --custom_vocabulary {custom_vocabulary}"""
            print("Custom vocabulary not supported for YOLO")

        self.num_sem_categories = 80

    def reset_vocab(self, new_vocab: List[str], vocab_type="custom"):
        print("Resetting vocabulary not supported for YOLO")

    def predict(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
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
        pred = self.yolo(image)
        task_observations = dict()

        if pred[0].boxes is None or pred[0].masks is None:
            task_observations["semantic_frame"] = None
            return (
                np.zeros((rgb.shape[0], rgb.shape[1])),
                -1 * np.ones((rgb.shape[0], rgb.shape[1])),
                task_observations,
            )

        class_idcs = pred[0].boxes.cls.cpu().numpy()
        masks = pred[0].masks.data.cpu().numpy()

        # Resize masks to original image size
        masks = np.array([cv2.resize(mask, (width, height)) for mask in masks])

        # Add some visualization code for YOLO
        if draw_instance_predictions:
            task_observations["semantic_frame"] = pred[0].plot(conf=False, labels=False, masks=True)
        else:
            task_observations["semantic_frame"] = None

        # Sort instances by mask size
        scores = pred[0].boxes.conf.cpu().numpy()

        if depth_threshold is not None and depth is not None:
            masks = np.array([filter_depth(mask, depth, depth_threshold) for mask in masks])

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))

        semantic = semantic_map.astype(int)
        instance = instance_map.astype(int)
        task_observations["instance_map"] = instance_map
        task_observations["instance_classes"] = class_idcs
        task_observations["instance_scores"] = scores

        return semantic, instance, task_observations


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
