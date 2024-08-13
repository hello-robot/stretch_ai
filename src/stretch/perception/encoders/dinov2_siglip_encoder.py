from typing import Optional

import clip
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from .base_encoder import BaseImageTextEncoder
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    SiglipModel,
    AutoTokenizer,
    Dinov2Model,
)
import requests


class Dinov2SigLIPEncoder(BaseImageTextEncoder):
    """Simple wrapper for encoding different things as text."""

    def __init__(
        self, version="google/siglip-base-patch16-224", device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.version = version
        self.model = AutoModel.from_pretrained(version).to(device)
        self.processor = AutoProcessor.from_pretrained(version)
        # self.visual_feat_encoder = DINOv2()
        self.tokenizer = AutoTokenizer.from_pretrained(version)

        self.visual_feat_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base"
        )
        self.visual_feat_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(
            device
        )

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
        inputs = self.tokenizer([text], padding="max_length", return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.float()

    def get_visual_feat(self, image: np.ndarray):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy() * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        inputs = self.visual_feat_processor(images=pil_image, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.visual_feat_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).float()


# class DINOv2(torch.nn.Module):
#     """Runs DINOv2 and upsamples either with featup or with interpolation"""

#     def __init__(self) -> None:
#         super().__init__()
#         self.build_model()

#     def build_model(self) -> None:
#         self.patch_size = 14
#         self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
#         self.transform = transforms.Compose(
#             [
#                 lambda x: 255.0 * x,
#                 transforms.Normalize(
#                     mean=[123.675, 116.28, 103.53],
#                     std=[58.395, 57.12, 57.375],
#                 ),
#             ]
#         )

#     def forward(self, image: torch.TensorType) -> torch.TensorType:
#         """Encodes the image with DINOv2, then uses FeatUp to maintain the input resolution

#         Args:
#             image (torch.Tensor): (B, 3, H, W) batch of input images
#                 - Normalized to between 0 and 1

#         Returns:
#             torch.Tensor: DINOv2 features of the image at the original resolution.
#         """
#         original_shape = image.shape
#         image = self.transform(image)
#         image = torch.nn.functional.interpolate(
#             image,
#             (
#                 (image.shape[2] // self.patch_size) * self.patch_size,
#                 (image.shape[3] // self.patch_size) * self.patch_size,
#             ),
#         )
#         output = self.dino_model.forward_features(image)
#         feats = output["x_norm_patchtokens"]
#         feats = feats.reshape(
#             feats.shape[0],
#             image.shape[-2] // self.patch_size,
#             image.shape[-1] // self.patch_size,
#             feats.shape[-1],
#         )
#         feats = feats.permute(0, 3, 1, 2)
#         feats = torch.nn.functional.interpolate(
#             feats, original_shape[2:], mode="bilinear"
#         )
#         return feats