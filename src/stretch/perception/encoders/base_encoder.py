# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import abc

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from numpy import ndarray
from torch import Tensor


class BaseImageTextEncoder(abc.ABC):
    """
    Encodes images, encodes text, and allows comparisons between the two encoding.
    """

    @abc.abstractmethod
    def encode_image(self, image: Union[ndarray, Tensor]):
        raise NotImplementedError

    @abc.abstractmethod
    def encode_text(self, text: str):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_score(self, image: Tensor, text: Tensor):
        raise NotImplementedError
