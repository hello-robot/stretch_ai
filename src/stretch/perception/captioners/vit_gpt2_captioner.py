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
from overrides import override
from PIL import Image
from torch import Tensor
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

from .base_captioner import BaseCaptioner


class VitGPT2Captioner(BaseCaptioner):
    """Image captioner using Vision Transformer and GPT-2 model."""

    def __init__(self, max_length: int = 16, num_beams: int = 4, device: Optional[str] = None):
        """Initialize the ViT-GPT2 image captioner.

        Args:
            max_length (int, optional): Maximum length of the generated caption. Defaults to 16.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.
        """
        super(VitGPT2Captioner, self).__init__()
        self.max_length = max_length
        self.num_beams = num_beams
        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        # Create models
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        ).to(self._device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    @override
    def caption_image(self, image: Union[ndarray, Tensor, Image.Image]) -> str:
        """Generate a caption for the given image.

        Args:
            image (Union[ndarray, Tensor]): Image to generate caption for.

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

        pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        # Generate caption
        output_ids = self.model.generate(
            pixel_values,
            max_length=self.max_length,
            num_beams=self.num_beams,
            use_cache=True,
            no_repeat_ngram_size=3,
            do_sample=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
        )

        # Decode the output ids to text
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption


@click.command()
@click.option("--image_path", default="object.png", help="Path to image file")
def main(image_path: str):
    captioner = VitGPT2Captioner()

    # Load image from file
    image = Image.open(image_path)

    # Generate caption
    caption = captioner.caption_image(image)

    # Print caption
    print(caption)


if __name__ == "__main__":
    main()
