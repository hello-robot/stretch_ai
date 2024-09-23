# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


import math

import numpy as np

from stretch.agent.base import ManagedOperation


class NavigateToObjectOperation(ManagedOperation):

    plan = None
    for_manipulation: bool = True
    be_precise: bool = False

    def __init__(self, *args, to_receptacle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_receptacle = to_receptacle

    def get_target(self):
        if self.to_receptacle:
            return self.agent.current_receptacle
        else:
            return self.agent.current_object

    def can_start(self):
        print(
            f"{self.name}: check to see if object is reachable (receptacle={self.to_receptacle})."
        )
        self.plan = None
        if self.get_target() is None:
            self.error("no target!")
            return False

        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            self.error(
                "Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )
            breakpoint()

        if self.agent.within_reach_of(self.get_target()):
            self.cheer("Already within reach of object!")

        # Motion plan to the object
        plan = self.agent.plan_to_instance_for_manipulation(self.get_target(), start=start)

        if plan.success:
            self.plan = plan
            self.cheer("Found plan to object!")
            return True
        else:
            self.agent.set_instance_as_unreachable(self.get_target())
        self.error("Planning failed!")
        return False

    def run(self):
        """Navigate the robot to the object. If the object is within reach, we will not move the robot, but will rotate it to face the object.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """

        self.intro("executing motion plan to the object.")
        self.robot.move_to_nav_posture()

        if self.agent.within_reach_of(self.get_target()):
            self.warn("Already within reach of object!")
            xyz = self.get_target().get_center()
            start_xyz = self.robot.get_base_pose()[:2]
            theta = math.atan2(
                xyz[1] - self.robot.get_base_pose()[1], xyz[0] - self.robot.get_base_pose()[0]
            )
            self.robot.navigate_to(
                np.array([start_xyz[0], start_xyz[1], theta + np.pi / 2]),
                blocking=True,
                timeout=30.0,
            )
            return

        # Execute the trajectory
        assert (
            self.plan is not None
        ), "Did you make sure that we had a plan? You should call can_start() before run()."
        self.robot.execute_trajectory(self.plan, final_timeout=10.0)

        # Orient the robot towards the object and use the end effector camera to pick it up
        xyt = self.plan.trajectory[-1].state
        # self.robot.navigate_to(xyt + np.array([0, 0, np.pi / 2]), blocking=True, timeout=30.0)
        if self.be_precise:
            self.warn("Moving again to make sure we're close enough to the goal.")
            self.robot.navigate_to(xyt, blocking=True, timeout=30.0)

    def was_successful(self):
        """This will be successful if we got within a reasonable distance of the target object."""
        return True  # self.robot.in_navigation_mode()
