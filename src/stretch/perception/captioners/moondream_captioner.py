# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Union

import click
from numpy import ndarray
from overrides import override
from PIL import Image
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_captioner import BaseCaptioner

model_id = "vikhyatk/moondream2"
revision = "2024-07-23"


class MoondreamCaptioner(BaseCaptioner):
    def __init__(self):
        super(MoondreamCaptioner, self).__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    @override
    def caption_image(self, image: Union[ndarray, Tensor, Image.Image]) -> str:
        """Generate a caption for the given image.

        Args:
            image (Union[ndarray, Tensor, Image.Image]): Image to generate caption for.

        Returns:
            str: Generated caption.
        """
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            if isinstance(image, Tensor):
                _image = image.cpu().numpy()
            else:
                _image = image
            pil_image = Image.fromarray(_image)
        enc_image = self.model.encode_image(pil_image)
        return self.model.answer_question(enc_image, "Describe this image.", self.tokenizer)


@click.command()
@click.option("--image_path", default="object.png", help="Path to image file")
def main(image_path: str):
    captioner = MoondreamCaptioner()

    # Load image from file
    image = Image.open(image_path)

    # Generate caption
    caption = captioner.caption_image(image)

    # Print caption
    print(caption)


if __name__ == "__main__":
    main()
