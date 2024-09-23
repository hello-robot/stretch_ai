# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import numpy as np

from stretch.agent.base import ManagedOperation
from stretch.motion import HelloStretchIdx
from stretch.utils.geometry import point_global_to_base


class PreGraspObjectOperation(ManagedOperation):
    """Move the robot to a position looking at the object using the navigation/manipulation camera."""

    plan = None
    show_object_in_voxel_grid: bool = False
    use_pitch_from_vertical: bool = True
    grasp_distance_threshold: float = 0.8

    def can_start(self):
        """Can only move to an object if it's been picked out and is reachable."""

        self.plan = None
        if self.agent.current_object is None:
            return False
        elif self.agent.is_instance_unreachable(self.agent.current_object):
            return False

        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            self.error(
                f"{self.name}: [ERROR]: Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )
            breakpoint()

        # Get the center of the object point cloud so that we can look at it
        object_xyz = self.agent.current_object.point_cloud.mean(axis=0)
        dist = np.linalg.norm(object_xyz[:2] - start[:2])
        if dist > self.grasp_distance_threshold:
            self.error(f"Object is too far away to grasp: {dist}")
            return False
        self.cheer(f"{self.name}: Object is probably close enough to grasp: {dist}")
        return True

    def run(self):

        self.intro("Moving to a position to grasp the object.")
        self.robot.move_to_manip_posture()

        # Now we should be able to see the object if we orient gripper properly
        # Get the end effector pose
        obs = self.robot.get_observation()
        joint_state = obs.joint
        model = self.robot.get_robot_model()

        # Note that these are in the robot's current coordinate frame; they're not global coordinates, so this is ok to use to compute motions.
        ee_pos, ee_rot = model.manip_fk(joint_state)

        # Get the center of the object point cloud so that we can look at it
        object_xyz = self.agent.current_object.point_cloud.mean(axis=0)
        xyt = self.robot.get_base_pose()
        if self.show_object_in_voxel_grid:
            # Show where the object is together with the robot base
            self.agent.voxel_map.show(
                orig=object_xyz.cpu().numpy(), xyt=xyt, footprint=self.robot_model.get_footprint()
            )
        relative_object_xyz = point_global_to_base(object_xyz, xyt)

        # Compute the angles necessary
        if self.use_pitch_from_vertical:
            # dy = relative_gripper_xyz[1] - relative_object_xyz[1]
            dy = np.abs(ee_pos[1] - relative_object_xyz[1])
            dz = np.abs(ee_pos[2] - relative_object_xyz[2])
            pitch_from_vertical = np.arctan2(dy, dz)
            # current_ee_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
        else:
            pitch_from_vertical = 0.0

        # Joint state goal
        joint_state[HelloStretchIdx.WRIST_PITCH] = -np.pi / 2 + pitch_from_vertical

        # Strip out fields from the full robot state to only get the 6dof manipulator state
        # TODO: we should probably handle this in the zmq wrapper.
        # arm_cmd = conversions.config_to_manip_command(joint_state)
        self.robot.switch_to_manipulation_mode()
        self.robot.arm_to(joint_state, blocking=True)

    def was_successful(self):
        return self.robot.in_manipulation_mode()
