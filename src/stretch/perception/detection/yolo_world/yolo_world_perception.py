# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import copy
import os
import pathlib
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from stretcg.perception.constants import yolo_world_vocabulary
from torchvision.ops import nms

from stretch.core.abstract_perception import PerceptionModule
from stretch.core.interfaces import Observations
from stretch.perception.detection.utils import filter_depth, overlay_masks
from stretch.utils.config import get_full_config_path, load_config

sys.path.append(str(Path(__file__).resolve().parent / "YoloWorld"))


class YoloWorldPerception(PerceptionModule):
    def __init__(
        self,
        config_file=None,
        custom_vocabulary="",
        checkpoint_file=None,
        sem_gpu_id=0,
        verbose: bool = False,
        confidence_threshold: Optional[float] = None,
    ):
        """Load trained Detic model for inference.

        Arguments:
            config_file: path to model config
            # vocabulary: currently one of "coco" for indoor coco categories or "custom"
            #  for a custom set of categories
            custom_vocabulary: if vocabulary="custom", this should be a comma-separated
             list of classes (as a single string)
            checkpoint_file: path to model checkpoint
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
        """
        self.verbose = verbose
        if config_file is None:
            config_file = str(
                Path(__file__).resolve().parent / "YoloWorld/configs/segmentation/"
                "yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis.py"
            )
        else:
            config_file = get_full_config_path(config_file)

        if checkpoint_file is None:
            checkpoint_file = get_full_config_path(
                "perception/yolo_world_seg_m_dual_vlpan_2e-4_80e_8gpus_allmodules_finetune_lvis-ca465825.pth"
            )
        else:
            checkpoint_file = get_full_config_path(checkpoint_file)

        # Check if the checkpoint file exists
        if not Path(checkpoint_file).exists():
            # Download the checkpoint file
            os.system(
                f"wget -O {checkpoint_file} https://huggingface.co/wondervictor/YOLO-World/"
                "resolve/main/yolo_world_seg_m_dual_vlpan_2e-4"
                "_80e_8gpus_allmodules_finetune_lvis-ca465825.pth"
            )

        if self.verbose:
            print(f"Loading Yolo World with config={config_file} and checkpoint={checkpoint_file}")

        if sem_gpu_id == -1:
            raise NotImplementedError("Yolo World does not support running on CPU")

        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        self.mask_annotator = sv.MaskAnnotator()

        print(f"Loadingconfig from {config_file}")
        self.config = Config.fromfile(config_file)
        self.config.work_dir = "."
        self.config.load_from = checkpoint_file

        self.runner = Runner.from_cfg(self.config)
        self.runner.call_hook("before_run")
        self.runner.load_or_resume()
        self.config.test_dataloader.dataset.pipeline[0].type = "mmdet.LoadImageFromNDArray"
        pipeline = self.config.test_dataloader.dataset.pipeline
        self.runner.pipeline = Compose(pipeline)

        self.runner.model.eval()

        if custom_vocabulary != "":
            self.class_names = custom_vocabulary.split(",")
        else:
            self.class_names = yolo_world_vocabulary

        self.instance_mode = ColorMode.IMAGE

    def reset_vocab(self, new_vocab: List[str], vocab_type="custom"):
        """Resets the vocabulary of Detic model allowing you to change detection on
        the fly. Note that previous vocabulary is not preserved.
        Args:
            new_vocab: list of strings representing the new vocabulary
            vocab_type: one of "custom" or "coco"; only "custom" supported right now
        """
        if self.verbose:
            print(f"Resetting vocabulary to {new_vocab}")
        if "__unused" in MetadataCatalog.keys():
            MetadataCatalog.remove("__unused")
        if vocab_type == "custom":
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = new_vocab
            # classifier = get_clip_embeddings(self.metadata.thing_classes)
            self.categories_mapping = {i: i for i in range(len(self.metadata.thing_classes))}
        else:
            raise NotImplementedError(
                "Yolo does not have support for resetting from custom to coco vocab"
            )
        self.num_sem_categories = len(self.categories_mapping)

        num_classes = len(self.metadata.thing_classes)
        self.tempfile = NamedTemporaryFile(suffix=".png")

    def predict(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order - Detic expects BGR)
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

        # Save the image to a temporary file
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        data_info = self.runner.pipeline(
            dict(img_id=0, img=image, texts=[[t.strip()] for t in self.class_names + [" "]])
        )
        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0), data_samples=[data_info["data_samples"]]
        )

        with autocast(enabled=False), torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        # nms
        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=0.5)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > 0.05]

        if len(pred_instances.scores) > 100:
            indices = pred_instances.scores.float().topk(100)[1]
            pred_instances = pred_instances[indices]

        # predictions
        pred_instances = pred_instances.cpu().numpy()

        if "masks" in pred_instances:
            masks = pred_instances["masks"]
        else:
            masks = None

        detections = sv.Detections(
            xyxy=pred_instances["bboxes"],
            class_id=pred_instances["labels"],
            confidence=pred_instances["scores"],
        )

        # label ids with confidence scores
        labels = [
            f"{class_id} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        task_observations = dict()

        # Add some visualization code for Detic
        if draw_instance_predictions:
            visualization = self.mask_annotator.annotate(image, detections)
            task_observations["semantic_frame"] = visualization
        else:
            task_observations["semantic_frame"] = None

        task_observations["instance_classes"] = detections.class_id
        task_observations["instance_scores"] = detections.confidence
        task_observations["bounding_boxes"] = detections.xyxy

        return None, None, task_observations


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
