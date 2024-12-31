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

from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection

from stretch.utils.image import Camera, camera_xyz_to_global_xyz


class OwlPerception:
    def __init__(
        self,
        version="owlv2-L-p14-ensemble",
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = 0.2,
    ):
        """Load trained OWL model for inference.

        Arguments:
            version: owlv2 version, currently supporting google/owlv2-large-patch14-ensemble and google/owlv2-base-patch16-ensemble
            device: which device you want to run the model,
        """
        self.confidence_threshold = confidence_threshold
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if version == "owlv2-L-p14-ensemble":
            configuration = "google/owlv2-large-patch14-ensemble"
        elif version == "owlv2-B-p16-ensemble":
            configuration = "google/owlv2-base-patch16-ensemble"
        elif version == "owlv2-B-p16":
            configuration = "google/owlv2-base-patch16"
        else:
            raise ValueError("Owlv2 version not implemented yet!")

        self.processor = AutoProcessor.from_pretrained(configuration)
        self.model = Owlv2ForObjectDetection.from_pretrained(configuration).to(self.device)

        print(f"Loaded owl model from {configuration}")

    def detect_object(
        self,
        rgb: torch.Tensor,
        text: str,
        confidence_threshold: Optional[float] = None,
    ):
        """Try to find target objects given one text query.
        Arguments:
            rgb: ideally of shape (C, H, W), the pixel value should be integer between [0, 255]
        """
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.Tensor(rgb).to(torch.uint8)
        rgb = rgb.squeeze()
        if rgb.shape[0] != 3:
            rgb = rgb.permute(2, 0, 1)
        inputs = self.processor(text=[["a photo of a " + text]], images=rgb, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([rgb.size()[-2:]]).to(self.device)
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )[0]

        return results["scores"], results["boxes"]

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
            rgb=rgb, text=text, confidence_threshold=confidence_threshold
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
