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

from nav2_msgs.srv import LoadMap, SaveMap

from .abstract import AbstractControlModule


class StretchMappingClient(AbstractControlModule):
    def __init__(
        self,
        ros_client,
    ):
        super().__init__()

        self._ros_client = ros_client

        self._is_saved = False
        self._is_loaded = False

    # Interface methods
    def save_map(self, filename: str) -> bool:
        req = SaveMap.Request()
        req.map_url = filename
        result = self._ros_client.save_map_service.call(req)

        self._is_saved = result.result

        return self._is_saved

    def load_map(self, filename: str) -> bool:
        req = LoadMap.Request()
        req.map_url = filename
        result = self._ros_client.load_map_service.call(req)

        self._is_loaded = result.result

        return self._is_loaded

    def _enable_hook(self) -> bool:
        """Dummy override for abstract method"""

    def _disable_hook(self) -> bool:
        """Dummy override for abstract method"""
