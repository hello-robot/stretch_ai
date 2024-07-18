# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any

from .base_encoder import BaseImageTextEncoder
from .clip_encoder import ClipEncoder, NormalizedClipEncoder
from .siglip_encoder import SiglipEncoder


def get_encoder(encoder_name, args: Any):
    if encoder_name == "clip":
        return ClipEncoder(**args)
    elif encoder_name == "normalized_clip":
        return NormalizedClipEncoder(**args)
    elif encoder_name == "siglip":
        return SiglipEncoder(**args)
    else:
        raise ValueError(f"Encoder {encoder_name} not implemented or not supported.")
