# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional, Union

import click
import torch
from numpy import ndarray
from PIL import Image
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoProcessor

from .base_captioner import BaseCaptioner


class GitCaptioner(BaseCaptioner):
    """Image captioner using GIT (Generative Image-to-text Transformer) model."""

    def __init__(self, max_length: int = 30, num_beams: int = 4, device: Optional[str] = None):
        """Initialize the GIT image captioner.

        Args:
            max_length (int, optional): Maximum length of the generated caption. Defaults to 30.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.
            device (str, optional): Device to run the model on. Defaults to None (auto-detect).
        """
        super(GitCaptioner, self).__init__()
        self.max_length = max_length
        self.num_beams = num_beams
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Create models
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco").to(
            self._device
        )

    def caption_image(self, image: Union[ndarray, Tensor, Image.Image]) -> str:
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

        # Preprocess the image
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self._device)

        # Generate caption
        generated_ids = self.model.generate(
            pixel_values=inputs.pixel_values,
            max_length=self.max_length,
            num_beams=self.num_beams,
            do_sample=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
        )

        # Decode the output ids to text
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return caption


@click.command()
@click.option("--image_path", default="object.png", help="Path to image file")
def main(image_path: str):
    captioner = GitCaptioner()

    # Load image from file
    image = Image.open(image_path)

    # Generate caption
    caption = captioner.caption_image(image)

    # Print caption
    print(caption)


if __name__ == "__main__":
    main()
