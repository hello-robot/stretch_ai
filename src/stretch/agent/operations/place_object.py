# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

# Description: Place an object on top of the target receptacle, by just using the arm for now.
import time
from typing import Optional

import numpy as np

from stretch.agent.base import ManagedOperation
from stretch.motion import HelloStretchIdx
from stretch.utils.geometry import point_global_to_base


class PlaceObjectOperation(ManagedOperation):
    """Place an object on top of the target receptacle, by just using the arm for now."""

    place_distance_threshold: float = 0.8
    lift_distance: float = 0.2
    place_height_margin: float = 0.1
    show_place_in_voxel_grid: bool = False
    place_step_size: float = 0.15
    use_pitch_from_vertical: bool = True

    def get_target(self):
        return self.manager.current_receptacle

    def get_target_center(self):
        return self.get_target().point_cloud.mean(axis=0)

    def sample_placement_position(self, xyt) -> np.ndarray:
        """Sample a placement position for the object on the receptacle."""
        if self.get_target() is None:
            raise RuntimeError("no target set")
        target = self.get_target()
        center_xyz = self.get_target_center()
        print(" - Placing object on receptacle at", center_xyz)

        # Get the point cloud of the object and find distances to robot
        distances = (target.point_cloud[:, :2] - xyt[:2]).norm(dim=1)
        # Choose closest point to xyt
        idx = distances.argmin()
        # Get the point
        point = target.point_cloud[idx].cpu().numpy()
        print(" - Closest point to robot is", point)
        print(" - Distance to robot is", distances[idx])
        # Compute distance to the center of the object
        distance = np.linalg.norm(point[:2] - center_xyz[:2].cpu().numpy())
        # Take a step towards the center of the object
        dxyz = (center_xyz - point).cpu().numpy()
        point[:2] = point[:2] + (
            dxyz[:2] / np.linalg.norm(dxyz[:2]) * min(distance, self.place_step_size)
        )
        print(" - After taking a step towards the center of the object, we are at", point)
        return point

    def can_start(self) -> bool:
        self.attempt(
            "will start placing the object if we have object and receptacle, and are close enough to drop."
        )
        if self.manager.current_object is None or self.manager.current_receptacle is None:
            self.error("Object or receptacle not found.")
            return False
        object_xyz = self.get_target_center()
        start = self.robot.get_base_pose()
        dist = np.linalg.norm(object_xyz[:2] - start[:2])
        if dist > self.place_distance_threshold:
            self.error(f"Object is too far away to grasp: {dist}")
            return False
        self.cheer(f"Object is probably close enough to place upon: {dist}")
        return True

    def _get_place_joint_state(
        self, pos: np.ndarray, quat: np.ndarray, joint_state: Optional[np.ndarray] = None
    ):
        """Use inverse kinematics to compute joint position for (pos, quat) in base frame.

        Args:
            pos: 3D position of the target in the base frame
            quat: 4D quaternion of the target in the base frame
            joint_state: current joint state of the robot (optional) for inverse kinematics
        """
        if joint_state is None:
            joint_state = self.robot.get_observation().joint

        target_joint_positions, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            pos, quat, q0=joint_state
        )

        return target_joint_positions, success

    def run(self) -> None:
        self.intro("Placing the object on the receptacle.")
        self._successful = False

        # Get initial (carry) joint posture
        obs = self.robot.get_observation()
        joint_state = obs.joint
        model = self.robot.get_robot_model()

        # Switch to place position
        print(" - Move to manip posture")
        self.robot.move_to_manip_posture()

        # Get object xyz coords
        xyt = self.robot.get_base_pose()
        placement_xyz = self.sample_placement_position(xyt)
        print(" - Place object at", placement_xyz)

        # Get the center of the object point cloud so that we can place there
        relative_object_xyz = point_global_to_base(placement_xyz, xyt)

        # Compute the angles necessary
        if self.use_pitch_from_vertical:
            ee_pos, ee_rot = model.manip_fk(joint_state)
            # dy = relative_gripper_xyz[1] - relative_object_xyz[1]
            dy = np.abs(ee_pos[1] - relative_object_xyz[1])
            dz = np.abs(ee_pos[2] - relative_object_xyz[2])
            pitch_from_vertical = np.arctan2(dy, dz)
            # current_ee_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
        else:
            pitch_from_vertical = 0.0

        # Joint compute a joitn state goal and associated ee pos/rot
        joint_state[HelloStretchIdx.WRIST_PITCH] = -np.pi / 2 + pitch_from_vertical
        self.robot.arm_to(joint_state)
        ee_pos, ee_rot = model.manip_fk(joint_state)

        # Get max xyz
        max_xyz = self.get_target().point_cloud.max(axis=0)[0]

        # Placement is at xy = object_xyz[:2], z = max_xyz[2] + margin
        place_xyz = np.array(
            [relative_object_xyz[0], relative_object_xyz[1], max_xyz[2] + self.place_height_margin]
        )

        if self.show_place_in_voxel_grid:
            self.agent.voxel_map.show(
                orig=place_xyz, xyt=xyt, footprint=self.robot_model.get_footprint()
            )

        target_joint_positions, success = self._get_place_joint_state(
            pos=place_xyz, quat=ee_rot, joint_state=joint_state
        )
        self.attempt(f"Trying to place the object on the receptacle at {place_xyz}.")
        if not success:
            self.error("Could not place object!")
            return

        # Move to the target joint state
        self.robot.arm_to(target_joint_positions, blocking=True)
        time.sleep(0.5)

        # Open the gripper
        self.robot.open_gripper(blocking=True)
        time.sleep(0.5)

        # Move directly up
        target_joint_positions_lifted = target_joint_positions.copy()
        target_joint_positions_lifted[HelloStretchIdx.LIFT] += self.lift_distance
        self.robot.arm_to(target_joint_positions_lifted, blocking=True)

        # Return arm to initial configuration and switch to nav posture
        self.robot.move_to_nav_posture()
        self._successful = True

        self.cheer("We believe we successfully placed the object.")

    def was_successful(self):
        self.error("Success detection not implemented.")
        return self._successful
