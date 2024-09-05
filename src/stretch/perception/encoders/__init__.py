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
from typing import Any

from .base_encoder import BaseImageTextEncoder
from .clip_encoder import ClipEncoder, NormalizedClipEncoder
from .dinov2_siglip_encoder import Dinov2SigLIPEncoder
from .siglip_encoder import SiglipEncoder

encoders = ["clip", "normalized_clip", "siglip", "dinov2siglip"]


def get_encoder(encoder_name, args: Any):
    if encoder_name == "clip":
        return ClipEncoder(**args)
    elif encoder_name == "normalized_clip":
        return NormalizedClipEncoder(**args)
    elif encoder_name == "siglip":
        return SiglipEncoder(**args)
    elif encoder_name == "dinov2siglip":
        return Dinov2SigLIPEncoder(**args)
    else:
        raise ValueError(f"Encoder {encoder_name} not implemented or not supported.")
