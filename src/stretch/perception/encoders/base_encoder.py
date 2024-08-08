# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from numpy import ndarray
from torch import Tensor


class BaseImageTextEncoder:
    """
    Encodes images, encodes text, and allows comparisons between the two encoding.
    """

    def encode_image(self, image: Union[ndarray, Tensor]):
        raise NotImplementedError

    def encode_text(self, text: str):
        raise NotImplementedError
