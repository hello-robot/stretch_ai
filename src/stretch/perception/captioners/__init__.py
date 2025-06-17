# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
from typing import Any

from .base_captioner import BaseCaptioner
from .git_captioner import GitCaptioner
from .qwen_captioner import QwenCaptioner
from .vit_gpt2_captioner import VitGPT2Captioner

captioners = ["git", "vit_gpt2", "qwen"]


def get_captioner(captioner_name, args: Any) -> BaseCaptioner:
    """Get captioner."""
    if captioner_name == "git":
        return GitCaptioner(**args)
    elif captioner_name == "vit_gpt2":
        return VitGPT2Captioner(**args)
    else:
        raise ValueError(
            f"Captioner {captioner_name} not implemented or not supported. Should be one of {captioners}."
        )
