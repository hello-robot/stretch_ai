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
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor


class PaligemmaCaptioner:
    """Image captioner using Paligemma2 model."""

    def __init__(
        self,
        max_length: int = 100,
        num_beams: int = 1,
        device: Optional[str] = None,
        image_shape=None,
    ):
        """Initialize the Paligemma2 image captioner.

        Args:
            max_length (int, optional): Maximum length of the generated caption. Defaults to 100.
            num_beams (int, optional): Number of beams for beam search. Defaults to 1.
            device (str, optional): Device to run the model on. Defaults to None (auto-detect).

        TODO: Integrate other Paligemma2 versions, for now it supports 7B so that the model is good enough while not too large.
        """
        self.max_length = max_length
        self.num_beams = num_beams
        self.image_shape = image_shape
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        model_id = "google/paligemma2-3b-pt-448"
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=self._device
        ).eval()
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)

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
            bbox[2] = max(h - 2, bbox[2])
            bbox[3] = max(w - 2, bbox[3])
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
            prompt = "Describe the object in the red bounding box."
        prompt = "<image>" + prompt + "Include as many details as possible!"

        model_inputs = (
            self.processor(text=prompt, images=pil_image, return_tensors="pt")
            .to(torch.bfloat16)
            .to(self.model.device)
        )
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=self.max_length, do_sample=False
            )
            generation = generation[0][input_len:]
            output_text = self.processor.decode(generation, skip_special_tokens=True)

        return output_text
