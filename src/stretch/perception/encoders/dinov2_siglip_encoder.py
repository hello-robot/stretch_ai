# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer, Dinov2Model

from .base_encoder import BaseImageTextEncoder


class Dinov2SigLIPEncoder(BaseImageTextEncoder):
    """Simple wrapper for encoding different things as text."""

    def __init__(
        self, version="google/siglip-base-patch16-224", device: Optional[str] = None, **kwargs
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.version = version
        self.model = AutoModel.from_pretrained(version).to(device)
        self.processor = AutoProcessor.from_pretrained(version)
        # self.visual_feat_encoder = DINOv2()
        self.tokenizer = AutoTokenizer.from_pretrained(version)

        self.visual_feat_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.visual_feat_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)

    def encode_image(self, image: np.ndarray):
        """Encode this input image to a sigclip vector"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy() * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.float()

    def encode_text(self, text: str):
        """Return clip vector for text"""
        inputs = self.tokenizer([text], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.float()

    def get_visual_feat(self, image: np.ndarray):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy() * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        inputs = self.visual_feat_processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.visual_feat_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).float()

    def compute_score(self, text: str, image: np.ndarray):
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)
        return torch.nn.functional.cosine_similarity(text_features, image_features).item()
