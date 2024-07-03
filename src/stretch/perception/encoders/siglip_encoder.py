from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .base_encoder import BaseImageTextEncoder


class SiglipEncoder(BaseImageTextEncoder):
    def __init__(self, device: Optional[str] = None, **kwargs) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # self.processor = AutoProcessor.from_pretrained("salesforce/siglip-visual-encoder")
        # self.model = AutoModel.from_pretrained("salesforce/siglip-visual-encoder").to(self.device)
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    def encode_image(self, image: Union[torch.tensor, np.ndarray]) -> torch.Tensor:
        """Encode this input image to a CLIP vector"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy() * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model(**inputs).last_hidden_state
        breakpoint()
        return image_features.float()

    def encode_text(self, text: str) -> torch.Tensor:
        """Return clip vector for text"""
        inputs = self.processor(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model(**inputs).last_hidden_state
        breakpoint()
        return text_features.float()
