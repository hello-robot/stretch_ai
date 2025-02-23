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

from .owl_perception import OwlPerception


class OWLSAMProcessor(OwlPerception):
    def __init__(
        self,
        version="owlv2-L-p14-ensemble",
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = 0.2,
    ):
        """
        Marrying Owlv2 and Samv2 to complete open vocabulary segmentation.
        """
        super().__init__(version=version, device=device, confidence_threshold=confidence_threshold)
        # We assume that you use OWLSAM because you want to work on open vocab segmentation,
        # in this case, open vocab is the concentration

        # Considering SAM2 is not installed by default, we would not import it unless we need to use it.
        from stretch.perception.detection.sam2 import SAM2Perception

        self.sam_model = SAM2Perception(configuration="t")

    def detect_object(
        self,
        image: Type[Image.Image],
        text: str = None,
        box_filename: str = None,
        visualize_mask: bool = False,
        mask_filename: str = None,
    ) -> Tuple[np.ndarray, List[int]]:
        print("OWLSAM detection !!!")
        inputs = self.processor(text=[["a photo of a " + text]], images=image, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to("cuda")
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=0.05, target_sizes=target_sizes
        )[0]

        if len(results["boxes"]) == 0:
            return None, None

        bounding_box = results["boxes"][torch.argmax(results["scores"])]

        bounding_boxes = bounding_box.unsqueeze(0)

        # self.mask_predictor.set_image(image)
        # masks, _, _ = self.mask_predictor.predict(
        #     point_coords=None, point_labels=None, box=bounding_boxes, multimask_output=False
        # )
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
