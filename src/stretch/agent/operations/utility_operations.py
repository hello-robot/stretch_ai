# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from stretch.agent.base import ManagedOperation


class SetCurrentObjectOperation(ManagedOperation):
    """Set the current object for manipulation."""

    def __init__(self, *args, target=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target

    def can_start(self):
        print(f"{self.name}: setting target object to {self.target}.")
        if self.target is None:
            self.error("no target object!")
            return False
        return True

    def run(self):
        self.agent.current_object = self.target

    def was_successful(self):
        return self.agent.current_object == self.target


class SetCurrentReceptacleOperation(ManagedOperation):
    """Set the current receptacle for manipulation."""

    def __init__(self, *args, target=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target

    def can_start(self):
        print(f"{self.name}: setting target receptacle to {self.target}.")
        if self.target is None:
            self.error("no target receptacle!")
            return False
        return True

    def run(self):
        self.agent.current_receptacle = self.target

    def was_successful(self):
        return self.agent.current_receptacle == self.target
