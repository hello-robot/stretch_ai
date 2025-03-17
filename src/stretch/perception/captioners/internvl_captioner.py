# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional, Union

import torch
import torchvision.transforms as T
from numpy import ndarray
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(pil_image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternvlCaptioner:
    """Image captioner using Internvl 2.5 model."""

    def __init__(
        self,
        device: Optional[str] = None,
        image_shape=None,
    ):
        """Initialize the BLIP image captioner.

        Args:
            max_length (int, optional): Maximum length of the generated caption. Defaults to 30.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.
            device (str, optional): Device to run the model on. Defaults to None (auto-detect).
        """
        self.image_shape = image_shape
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Create models
        path = "OpenGVLab/InternVL2_5-8B"
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    def caption_image(
        self,
        image: Union[ndarray, Tensor, Image.Image],
        bbox: Optional[Union[list, Tensor, ndarray]] = None,
    ) -> str:
        """Generate a caption for the given image.

        Args:
            image (Union[ndarray, Tensor, Image.Image]): The input image.

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

        if bbox is not None:
            h, w = pil_image.size
            bbox[0] = max(1, bbox[0])
            bbox[1] = max(1, bbox[1])
            bbox[2] = max(h - 2, bbox[2])
            bbox[3] = max(w - 2, bbox[3])
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle(bbox, outline="red", width=1)
        if self.image_shape is not None:
            pil_image = pil_image.resize(self.image_shape)

        if bbox is None:
            prompt = "Describe the image."
        else:
            prompt = "Describe the object in the red bounding box."

        question = (
            "<image>\n"
            + prompt
            + "\nLimit your answer in 10 words. E.G. a yellow banana; a white hand sanitizer."
        )
        pixel_values = load_image(pil_image, max_num=12).to(torch.bfloat16).to(self._device)
        generation_config = dict(max_new_tokens=100, do_sample=True)
        output_text = self.model.chat(self.tokenizer, pixel_values, question, generation_config)

        return output_text
