# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from stretch.agent.base import ManagedOperation


class GoToOperation(ManagedOperation):
    """Put the robot into navigation mode"""

    location = None
    _successful = False

    def configure(self, location: str = ""):
        self.location = location

    def can_start(self) -> bool:
        return self.location is not None and self.location != ""

    def run(self) -> None:
        self.intro(f"Attempting move to {self.location}")
        _, instance = self.agent.get_instance_from_text(self.location)

        if instance is None:
            self.error(f"Could not find a matching instance to {self.location}!")
        self.agent.move_to_instance(instance)
        self._successful = True
        self.cheer(f"Done attempting moving to {self.location}")

    def was_successful(self) -> bool:
        return self._successful
