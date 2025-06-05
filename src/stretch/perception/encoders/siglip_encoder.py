# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from stretch.utils.logger import Logger

from .base_encoder import BaseImageTextEncoder

logger = Logger(__name__)


class SiglipEncoder(BaseImageTextEncoder):
    """Image/text feature encoder using SIGLip model.

    Referencing the following paper: https://arxiv.org/abs/2303.15343

    From the HuggingFace implementation here: https://huggingface.co/docs/transformers/v4.42.0/en/model_doc/siglip

    Generally, these features are much better than OpenAI CLIP for open-vocabulary object detection.
    """

    def __init__(
        self,
        normalize: bool = True,
        device: Optional[str] = None,
        version: Optional[str] = None,
        feature_matching_threshold: float = 0.05,
        **kwargs,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.normalize = normalize
        self.feature_matching_threshold = feature_matching_threshold

        if version is None:
            version = "base"

        if version == "base":
            model_name = "google/siglip-base-patch16-224"
        elif version == "so400m":
            model_name = "google/siglip-so400m-patch14-384"
        else:
            raise ValueError(f"Invalid version {version}: must be one of 'base', 'so400m'")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode_image(
        self,
        image: Union[torch.tensor, np.ndarray],
        image_shape=(360, 270),
        verbose: bool = False,
    ) -> torch.Tensor:
        """Encode this input image to a feature vector"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = image.astype(np.uint8)

        # We should avoid using PIL image to allow parrelism

        # pil_image = Image.fromarray(image)
        # if verbose:
        #     logger.info("Encoding image", pil_image.size)
        # inputs = self.processor(images=pil_image, return_tensors="pt")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        if self.normalize:
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

    def encode_text(self, text: str) -> torch.Tensor:
        """Return feature vector for text"""
        # inputs = self.processor(text, return_tensors="pt")
        inputs = self.tokenizer([text], padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        if self.normalize:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    def classify(self, image: Union[np.ndarray, torch.Tensor], text: str) -> torch.Tensor:
        """Classify image and text"""

        # Convert image to PIL
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)

        # Process image and text
        inputs = self.processor(
            images=pil_image, text=text, return_tensors="pt", padding="max_length"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Evaluate model
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits_per_image
        probs = torch.sigmoid(logits)
        return probs

    def encode_batch_text(self, texts: List[str]) -> torch.Tensor:
        """Return feature vector for text"""
        # inputs = self.processor(text, return_tensors="pt")
        inputs = self.tokenizer(texts, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.float()

    def compute_score(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Compute similarity score between image and text"""
        # return torch.sigmoid((image @ text.T).sum(dim=-1))
        return torch.cosine_similarity(image, text, dim=-1)


class MaskSiglipEncoder(SiglipEncoder):
    def __init__(
        self,
        device: Optional[str] = None,
        version: Optional[str] = None,
        feature_matching_threshold: float = 0.12,
    ) -> None:
        """
        Extract pixel-wise features from SIGLip model
        """
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
            image_shape: desired output shape, tuple (H1, W1)
        Output:
            image: RGB image, shape [3, H1, W1]
            features: pixel-wise features, shape [H1, W1, 512]
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
        """
        Same as run_mask_siglip, but for multiple images
        """
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
