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

import torch.nn.functional as F
from torchvision import transforms

class MaskClipEncoder(ClipEncoder):
    def __init__(
        self,
        version="ViT-B/32", 
        device: Optional[str] = None
    ) -> None:
        super().__init__(
            device=device,
            version=version,
        )
        self.clip_model, self.clip_preprocess = self.model, self.preprocess

    def forward_one_block(self, resblocks, x):
        q, k, v = None, None, None
        y = resblocks.ln_1(x)
        y = F.linear(y, resblocks.attn.in_proj_weight, resblocks.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, resblocks.attn.out_proj.weight, resblocks.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v = v + resblocks.mlp(resblocks.ln_2(v))

        return v

    def extract_mask_siglip_features(self, x, image_shape):
        with torch.no_grad():
                x = self.clip_model.visual.conv1(x)
                N, L, H, W = x.shape
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
                x = torch.cat(
                    [
                        self.clip_model.visual.class_embedding.to(x.dtype)
                        + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                        x,
                    ],
                    dim=1,
                )
                x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
                x = self.clip_model.visual.ln_pre(x)
                x = x.permute(1, 0, 2)
                for idx in range(self.clip_model.visual.transformer.layers):
                    if idx == self.clip_model.visual.transformer.layers - 1:
                        break
                    x = self.clip_model.visual.transformer.resblocks[idx](x)
                x = self.forward_one_block(self.clip_model.visual.transformer.resblocks[-1], x)
                x = x[1:]
                x = x.permute(1, 0, 2)
                x = self.clip_model.visual.ln_post(x)
                x = x @ self.clip_model.visual.proj
                feat = x.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        feat = F.interpolate(feat, image_shape, mode="bilinear", align_corners=True)
        feat = F.normalize(feat, dim=1)
        return feat.permute(0, 2, 3, 1)

    def run_mask_siglip(self, image, image_shape):
        """
        Run mask siglip
        Input:
            image: RGB image, shape [3, H, W]
        """
        if self.device == "cpu":
                    input = (
                        self.clip_preprocess(transforms.ToPILImage()(image))
                        .unsqueeze(0)
                        .to(self.device)
                    )
        else:
                    input = (
                        self.clip_preprocess(transforms.ToPILImage()(image))
                        .unsqueeze(0)
                        .to(self.device)
                        .half()
                    )
        if image_shape is not None:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = F.interpolate(
                image, size=image_shape, mode="bilinear", align_corners=False
            ).squeeze()
        features = self.extract_mask_siglip_features(input, image.shape[-2:]).cpu()

        return image, features
