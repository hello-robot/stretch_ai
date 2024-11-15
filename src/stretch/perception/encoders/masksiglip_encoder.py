# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional

import torch
import torch.nn.functional as F

from .siglip_encoder import SiglipEncoder


class MaskSiglipEncoder(SiglipEncoder):
    def __init__(
        self,
        device: Optional[str] = None,
        version: Optional[str] = None,
        feature_matching_threshold: float = 0.12,
    ) -> None:
        super().__init__(
            normalize=True,
            device=device,
            version=version,
            feature_matching_threshold=feature_matching_threshold,
        )

    def forward_one_block_(self, resblocks, x):
        x = F.linear(x, resblocks.in_proj_weight, resblocks.in_proj_bias)
        N, L, C = x.shape
        x = x.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        x = F.linear(x, resblocks.out_proj.weight, resblocks.out_proj.bias)
        q, k, v = x.tensor_split(3, dim=0)

        return v

    def extract_mask_siglip_features(self, x, image_shape):
        with torch.no_grad():
            output = self.model.vision_model(x["pixel_values"], output_hidden_states=True)
        feat = output.last_hidden_state
        feat = self.forward_one_block_(self.model.vision_model.head.attention, feat)
        feat = self.model.vision_model.head.layernorm(feat)
        feat = feat + self.model.vision_model.head.mlp(feat)
        feat = feat.detach().cpu()
        with torch.no_grad():
            N, L, H, W = self.model.vision_model.embeddings.patch_embedding(x["pixel_values"]).shape
        feat = feat.reshape(N, H, W, L).permute(0, 3, 1, 2)
        feat = F.interpolate(feat, image_shape, mode="bilinear", align_corners=True)
        feat = F.normalize(feat, dim=1)
        return feat.permute(0, 2, 3, 1)

    def run_mask_siglip(self, image, image_shape):
        """
        Run mask siglip
        Input:
            image: RGB image, shape [3, H, W]
        """
        input = self.processor(images=image, padding="max_length", return_tensors="pt")
        for i in input:
            input[i] = input[i].to(self.device)
        if image_shape is not None:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = F.interpolate(
                image, size=image_shape, mode="bilinear", align_corners=False
            ).squeeze()
        features = self.extract_mask_siglip_features(input, image.shape[-2:]).cpu()

        return image, features

    def extract_per_pixel_features(self, x, image_shape):
        with torch.no_grad():
            output = self.model.vision_model(x["pixel_values"], output_hidden_states=True)
            feat = output.last_hidden_state
            feat = self.forward_one_block_(self.model.vision_model.head.attention, feat)
            feat = self.model.vision_model.head.layernorm(feat)
            feat = feat + self.model.vision_model.head.mlp(feat)
            feat = feat.detach().cpu()
            N, L, H, W = self.model.vision_model.embeddings.patch_embedding(x["pixel_values"]).shape
            feat = feat.reshape(N, H, W, L).permute(0, 3, 1, 2)
        features = []
        for f, size in zip(feat, image_shape):
            f = F.interpolate(f.unsqueeze(0), size, mode="bilinear", align_corners=True)[0]
            f = F.normalize(f, dim=0).permute(1, 2, 0)
            features.append(f.detach().cpu())
        return features
