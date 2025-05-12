# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from typing import Optional, Union

import torch
from numpy import ndarray
from PIL import Image, ImageDraw
from torch import Tensor

from stretch.llms.qwen_client import Qwen25VLClient

# pip install flash-attn


class QwenCaptioner:
    """Image captioner using Qwen2.5 model."""

    def __init__(
        self,
        model_size: str = "3B",
        max_length: int = 200,
        num_beams: int = 1,
        device: Optional[str] = None,
        image_shape=None,
        draw_on_image=True,
    ):
        """Initialize the Qwen2.5 image captioner.

        Args:
            max_length (int, optional): Maximum length of the generated caption. Defaults to 100.
            num_beams (int, optional): Number of beams for beam search. Defaults to 1.
            device (str, optional): Device to run the model on. Defaults to None (auto-detect).

        TODO: Integrate other QwenVL2.5 versions, for now it supports 7B so that the model is good enough while not too large.
        """
        self.max_length = max_length
        self.num_beams = num_beams
        self.image_shape = image_shape
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.client = Qwen25VLClient(
            model_size=model_size, max_tokens=self.max_length, num_beams=self.num_beams
        )

        self.draw_on_image = draw_on_image

    def caption_image(
        self,
        image: Union[ndarray, Tensor, Image.Image],
        bbox: Optional[Union[list, Tensor, ndarray]] = None,
        verbose: bool = False,
    ) -> str:
        """Generate a caption for the given image.

        Args:
            image (Union[ndarray, Tensor, Image.Image]): The input image.
            bbox: Provide a bounding box if you just want to model to tell what is inside the box
            verbose: Set to True if you want to print some debug log

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

        if self.image_shape is not None:
            h, w = pil_image.size
            pil_image = pil_image.resize(self.image_shape)
            if bbox is not None:
                h1, w1 = self.image_shape
                bbox[0] = bbox[0] * h1 // h
                bbox[1] = bbox[1] * w1 // w
                bbox[2] = bbox[2] * h1 // h
                bbox[3] = bbox[3] * w1 // w
        if self.draw_on_image and bbox is not None:
            h, w = pil_image.size
            bbox[0] = max(1, bbox[0])
            bbox[1] = max(1, bbox[1])
            bbox[2] = min(h - 2, bbox[2])
            bbox[3] = min(w - 2, bbox[3])
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle(bbox, outline="red", width=1)

        if bbox is None:
            prompt = "Describe the image."
        elif self.draw_on_image:
            prompt = "Describe the object in the red bounding box."
        else:
            prompt = "Describe the object in the box " + str(bbox)

        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image",
        #                 "image": pil_image,
        #             },
        #             {"type": "text", "text": prompt},
        #             {
        #                 "type": "text",
        #                 "text": "Limit your answer in 10 words. E.G. a yellow banana; a white hand sanitizer",
        #             },
        #         ],
        #     }
        # ]

        messages = [
            pil_image,
            prompt,
            "Limit your answer in 10 words. E.G. a yellow banana; a white hand sanitizer",
        ]

        output_text = self.client(messages, verbose=verbose)

        if bbox is not None:
            if not self.draw_on_image:
                draw = ImageDraw.Draw(pil_image)
                draw.rectangle(bbox, outline="red", width=2)
            if not os.path.exists("test_caption/"):
                os.makedirs("test_caption")
            pil_image.save("test_caption/" + output_text + ".jpg")

        return output_text


if __name__ == "__main__":
    captioner = QwenCaptioner()
    caption = captioner.caption_image(Image.open("example.jpg"), verbose=True)
    print("caption for the image:", caption)
