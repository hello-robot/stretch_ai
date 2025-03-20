# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import copy
from typing import List, Optional, Tuple, Type

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from stretch.perception.detection.scannet_200_classes import CLASS_LABELS_200
from stretch.perception.detection.utils import filter_depth, overlay_masks

from .owl_perception import OwlPerception


class OWLSAMProcessor(OwlPerception):
    def __init__(
        self,
        version="owlv2-L-p14-ensemble",
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = 0.2,
        texts: Optional[List[str]] = None,
    ):
        """
        Marrying Owlv2 and Samv2 to complete open vocabulary segmentation.
        """
        super().__init__(version=version, device=device, confidence_threshold=confidence_threshold)
        # We assume that you use OWLSAM because you want to work on open vocab segmentation,
        # in this case, open vocab is the concentration

        self.reset_vocab(texts)

        # Considering SAM2 is not installed by default, we would not import it unless we need to use it.
        from stretch.perception.detection.sam2 import SAM2Perception

        self.sam_model = SAM2Perception(configuration="t")

    def reset_vocab(self, texts: Optional[List[str]] = None):
        if texts is None:
            texts = CLASS_LABELS_200
        self.texts = ["a photo of a " + text for text in texts]

    def predict(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
        confidence_threshold: Optional[float] = None,
    ):
        if isinstance(rgb, torch.Tensor):
            if rgb.ndim == 3 and rgb.shape[0] in [1, 3]:  # Tensor format: (C, H, W)
                rgb = rgb.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
            elif rgb.ndim == 4 and rgb.shape[0] == 1:  # Batched tensor: (B, C, H, W)
                rgb = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if isinstance(rgb, np.ndarray):
            if rgb.dtype != np.uint8:
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)

        height, width, _ = rgb.shape

        image = Image.fromarray(rgb)
        # Set default confidence threshold
        confidence_threshold = confidence_threshold or self.confidence_threshold

        # Process inputs for multiple text prompts
        inputs = self.processor(text=self.texts, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process detections
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )[0]

        # Early exit if no detections
        if len(results["boxes"]) == 0:
            return np.zeros_like(rgb[..., 0]), np.zeros_like(rgb[..., 0]), {}

        # Generate SAM masks for all detections
        boxes = results["boxes"]
        masks = self.sam_model.segment(image=image, xyxy=boxes)
        if len(masks) == 0:
            return np.zeros_like(rgb[..., 0]), np.zeros_like(rgb[..., 0]), {}
        mask = torch.Tensor(masks).bool()[0]

        # Process metadata
        class_idcs = results["labels"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        # Depth filtering
        if depth is not None and depth_threshold is not None:
            masks = np.array([filter_depth(mask, depth, depth_threshold) for mask in masks])

        # Sort instances by mask size (descending)
        mask_sizes = [np.sum(mask) for mask in masks]
        sorted_indices = np.argsort(mask_sizes)[::-1]

        # Reorder all data by mask size
        masks = [masks[i] for i in sorted_indices]
        class_idcs = class_idcs[sorted_indices]
        scores = scores[sorted_indices]

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))

        semantic = semantic_map.astype(int)
        instance = instance_map.astype(int)
        task_observations = dict()
        task_observations["instance_map"] = instance_map
        task_observations["instance_classes"] = class_idcs
        task_observations["instance_scores"] = scores

        return semantic, instance, task_observations

    def detect_object(
        self,
        image: Type[Image.Image],
        text: str = None,
        box_filename: str = None,
        visualize_mask: bool = False,
        mask_filename: str = None,
        threshold: Optional[float] = 0.05,
    ) -> Tuple[np.ndarray, List[int]]:
        inputs = self.processor(text=[["a photo of a " + text]], images=image, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        if threshold is None:
            threshold = self.confidence_threshold
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        if len(results["boxes"]) == 0:
            return None, None

        bounding_box = results["boxes"][torch.argmax(results["scores"])]

        bounding_boxes = bounding_box.unsqueeze(0)

        masks = self.sam_model.segment(image=image, xyxy=bounding_boxes)
        if len(masks) == 0:
            return None, None
        mask = torch.Tensor(masks).bool()[0]

        seg_mask = mask.detach().cpu().numpy()
        bbox = np.array(bounding_box.detach().cpu(), dtype=int)

        if visualize_mask:
            self.draw_bounding_box(image, bbox, box_filename)
            self.draw_mask_on_image(image, seg_mask, mask_filename)

        return seg_mask, bbox

    def draw_bounding_box(
        self, image: Type[Image.Image], bbox: List[int], save_file: str = None
    ) -> None:
        new_image = copy.deepcopy(image)
        draw_rectangle(new_image, bbox)

        if save_file is not None:
            new_image.save(save_file)

    def draw_mask_on_image(
        self, image: Type[Image.Image], seg_mask: np.ndarray, save_file: str = None
    ) -> None:
        image = np.array(image)
        image[seg_mask] = image[seg_mask] * 0.2

        # overlay mask
        highlighted_color = [179, 210, 255]
        overlay_mask = np.zeros_like(image)
        overlay_mask[seg_mask] = highlighted_color

        # placing mask over image
        alpha = 0.6
        highlighted_image = cv2.addWeighted(overlay_mask, alpha, image, 1, 0)
        highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_file, highlighted_image)
        print(f"Saved Segmentation Mask at {save_file}")

    def is_semantic(self):
        return True

    def is_instance(self):
        return True


def draw_rectangle(image, bbox, width=5):
    img_drw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    width_increase = 5
    for _ in range(width_increase):
        img_drw.rectangle([(x1, y1), (x2, y2)], outline="green")

        x1 -= 1
        y1 -= 1
        x2 += 1
        y2 += 1

    return img_drw
