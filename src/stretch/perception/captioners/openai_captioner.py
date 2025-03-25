# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import base64
from io import BytesIO
from typing import Optional, Union

import torch
from numpy import ndarray
from PIL import Image, ImageDraw
from torch import Tensor

from stretch.llms import OpenaiClient


class OpenaiCaptioner:
    """Image captioner using Qwen2 model."""

    def __init__(
        self,
        device: Optional[str] = None,
        image_shape=None,
        model_type: Optional[str] = "gpt-4o-mini",
    ):
        """Initialize the GPT image captioner.

        Args:
            device (str, optional): Device to run the model on. Defaults to None (auto-detect).
        """
        self.image_shape = image_shape
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Create client
        self.client = OpenaiClient(
            prompt=" \
            Limit your answer in 10 words. \
            E.G. a yellow banana; a white hand sanitizer ",
            model=model_type,
        )

    def caption_image(
        self,
        image: Union[ndarray, Tensor, Image.Image],
        bbox: Optional[Union[list, Tensor, ndarray]] = None,
    ) -> str:
        """Generate a caption for the given image.

        Args:
            image (Union[ndarray, Tensor, Image.Image]): The input image.
            bbox: Provide a bounding box if you just want to model to tell what is inside the box

        Returns:
            str: The generated caption.
        """
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            if isinstance(image, Tensor):
                _image = image.cpu().numpy()
            else:
                _image = image
            pil_image = Image.fromarray(_image)

        buffered = BytesIO()

        if bbox is not None:
            h, w = pil_image.size
            bbox[0] = max(1, bbox[0])
            bbox[1] = max(1, bbox[1])
            bbox[2] = min(h - 2, bbox[2])
            bbox[3] = min(w - 2, bbox[3])
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle(bbox, outline="red", width=2)
        if self.image_shape is not None:
            pil_image = pil_image.resize(self.image_shape)
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        base64_encoded = base64.b64encode(img_bytes).decode("utf-8")

        if bbox is None:
            prompt = "Describe the image."
        else:
            prompt = "Describe the object in the red box."

        command = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_encoded}",
                },
            },
        ]

        output_text = self.client(command)

        return output_text
