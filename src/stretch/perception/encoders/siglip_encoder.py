# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .base_encoder import BaseImageTextEncoder


class SiglipEncoder(BaseImageTextEncoder):
    """Image/text feature encoder using SIGLip model.

    Referencing the following paper: https://arxiv.org/abs/2303.15343

    From the HuggingFace implementation here: https://huggingface.co/docs/transformers/v4.42.0/en/model_doc/siglip

    Generally, these features are much better than OpenAI CLIP for open-vocabulary object detection.
    """

    def __init__(self, device: Optional[str] = None, **kwargs) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)

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
        return image_features.float()

    def encode_text(self, text: str) -> torch.Tensor:
        """Return feature vector for text"""
        # inputs = self.processor(text, return_tensors="pt")
        inputs = self.tokenizer([text], padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.float()
