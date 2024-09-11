# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Union

import clip
import numpy as np
import torch
from PIL import Image

from .base_encoder import BaseImageTextEncoder


class ClipEncoder(BaseImageTextEncoder):
    """Simple wrapper for encoding different things as text."""

    def __init__(self, version="ViT-B/32", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.version = version
        self.model, self.preprocess = clip.load(self.version, device=self.device)

    def encode_image(self, image: Union[torch.tensor, np.ndarray]) -> torch.Tensor:
        """Encode this input image to a CLIP vector"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy() * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
        return image_features.float()

    def encode_text(self, text: str) -> torch.Tensor:
        """Return clip vector for text"""
        text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features.float()

    def compute_score(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Compute similarity score between image and text"""
        return (100.0 * image @ text.T).squeeze()


class NormalizedClipEncoder(ClipEncoder):
    """Simple wrapper for encoding different things as text. Normalizes the results."""

    def encode_image(self, image: Union[torch.tensor, np.ndarray]) -> torch.Tensor:
        """Encode this input image to a CLIP vector"""
        image_features = super().encode_image(image)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def encode_text(self, text: str) -> torch.Tensor:
        """Return clip vector for text"""
        text_features = super().encode_text(text)
        return text_features / text_features.norm(dim=-1, keepdim=True)
