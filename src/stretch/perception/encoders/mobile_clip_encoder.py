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

import numpy as np
import open_clip
import torch
from mobileclip.modules.common.mobileone import reparameterize_model
from PIL import Image

from .base_encoder import BaseImageTextEncoder


class MobileClipEncoder(BaseImageTextEncoder):
    """Simple wrapper for encoding different things as text."""

    def __init__(self, version="S2", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        assert version in ["S1", "S2", "B"]
        self.version = "MobileCLIP-" + version

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.version, device=self.device, pretrained="datacompdr"
        )
        self.tokenizer = open_clip.get_tokenizer(self.version)
        self.model.eval()
        self.model = reparameterize_model(self.model)

    def encode_image(self, image: Union[torch.tensor, np.ndarray]) -> torch.Tensor:
        """Encode this input image to a CLIP vector"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy() * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
        image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

    def encode_text(self, text: str) -> torch.Tensor:
        """Return clip vector for text"""
        text = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    def compute_score(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Compute similarity score between image and text"""
        return (100.0 * image @ text.T).squeeze()


import torch.nn.functional as F
from torchvision import transforms


class MaskMobileClipEncoder(MobileClipEncoder):
    def __init__(self, version="S2", device: Optional[str] = None) -> None:
        super().__init__(
            device=device,
            version=version,
        )
        self.clip_model, self.clip_preprocess = self.model, self.preprocess
        if version != "B":
            # Instead of global pool all patch level features and project them with head
            self.clip_head = self.clip_model.visual.trunk.head
            self.clip_model.visual.trunk.head = torch.nn.Identity()

    def extract_mask_siglip_features(self, x, image_shape):
        with torch.no_grad():
            if self.version != "MobileCLIP-B":
                x = self.clip_model.visual(x)
                x = self.clip_head.fc(x.permute(0, 2, 3, 1))
            else:
                x = self.clip_model.visual.trunk.forward_features(x)[:, 1:]
                x = self.clip_model.visual.trunk.head(x)
                N, P, L = x.shape
                assert P == 196
                x = x.reshape(N, 14, 14, L)
            feat = x.permute(0, 3, 1, 2)
        feat = F.interpolate(feat, image_shape, mode="bilinear", align_corners=True)
        feat = F.normalize(feat, dim=1)
        return feat.permute(0, 2, 3, 1)

    def run_mask_siglip(self, image, image_shape):
        """
        Run mask siglip
        Input:
            image: RGB image, shape [3, H, W]
        """
        if not isinstance(image, torch.Tensor):
            image = torch.Tensor(image)
        if self.device == "cpu":
            input = (
                self.clip_preprocess(transforms.ToPILImage()(image)).unsqueeze(0).to(self.device)
            )
        else:
            input = (
                self.clip_preprocess(transforms.ToPILImage()(image)).unsqueeze(0).to(self.device)
            )
        if image_shape is not None:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = F.interpolate(
                image, size=image_shape, mode="bilinear", align_corners=False
            ).squeeze()
        features = self.extract_mask_siglip_features(input, image.shape[-2:]).cpu()

        return image, features
