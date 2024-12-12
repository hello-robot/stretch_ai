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
from PIL import Image

from .base_encoder import BaseImageTextEncoder


class CustomImageTextEncoder(BaseImageTextEncoder):
    """Image/text feature encoder using SIGLip model.

    Referencing the following paper: https://arxiv.org/abs/2303.15343

    From the HuggingFace implementation here: https://huggingface.co/docs/transformers/v4.42.0/en/model_doc/siglip

    Generally, these features are much better than OpenAI CLIP for open-vocabulary object detection.
    """

    def __init__(
        self,
        model,
        processor,
        tokenizer,
        normalize: bool = True,
        device: Optional[str] = None,
        **kwargs
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.normalize = normalize
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model

    def encode_image(self, image: Union[torch.tensor, np.ndarray]) -> torch.Tensor:
        """Encode this input image to a feature vector"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
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
