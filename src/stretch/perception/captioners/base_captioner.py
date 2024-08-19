# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import abc
from typing import Union

from numpy import ndarray
from torch import Tensor


class BaseImageCaptioner(abc.ABC):
    """Base class for image captioning models."""

    @abc.abstractmethod
    def caption_image(self, image: Union[ndarray, Tensor]):
        """Generate a caption for an image."""
        raise NotImplementedError
