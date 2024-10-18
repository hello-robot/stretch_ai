# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Description: Place an object on top of the target receptacle, by just using the arm for now.
import time
from typing import Optional

import numpy as np

from stretch.agent.base import ManagedOperation
from stretch.mapping.instance import Instance
from stretch.motion import HelloStretchIdx
from stretch.utils.geometry import point_global_to_base


class PlaceObjectOperation(ManagedOperation):
    """Place an object on top of the target receptacle, by just using the arm for now."""

    lift_distance: float = 0.2
    place_height_margin: float = 0.1
    show_place_in_voxel_grid: bool = False
    place_step_size: float = 0.35
    use_pitch_from_vertical: bool = True
    verbose: bool = True
    talk: bool = True

    def configure(
        self,
        lift_distance: float = 0.2,
        place_height_margin: float = 0.1,
        show_place_in_voxel_grid: bool = False,
        place_step_size: float = 0.25,
        use_pitch_from_vertical: bool = True,
    ):
        """Configure the place operation.

        Args:
            lift_distance: Distance to lift the object
            place_height_margin: Height margin for placing object
            show_place_in_voxel_grid: Show the place in voxel grid
            place_step_size: Step size for placing object. After finding closest point on target object to the robot, step this far towards object center.
            use_pitch_from_vertical: Use pitch from vertical
        """
        self.lift_distance = lift_distance
        self.place_height_margin = place_height_margin
        self.show_place_in_voxel_grid = show_place_in_voxel_grid
        self.place_step_size = place_step_size
        self.use_pitch_from_vertical = use_pitch_from_vertical

    def get_target(self) -> Instance:
        """Get the target object to place."""
        return self.agent.current_receptacle

    def get_target_center(self):
        return self.get_target().point_cloud.mean(axis=0)

    def sample_placement_position(self, xyt) -> np.ndarray:
        """Sample a placement position for the object on the receptacle."""
        if self.get_target() is None:
            raise RuntimeError("no target set")

        target = self.get_target()
        center_xyz = self.get_target_center()
        if self.verbose:
            print(" - Placing object on receptacle at", center_xyz)

        # Get the point cloud of the object and find distances to robot
        distances = (target.point_cloud[:, :2] - xyt[:2]).norm(dim=1)
        # Choose closest point to xyt
        idx = distances.argmin()
        # Get the point
        point = target.point_cloud[idx].cpu().numpy()
        if self.verbose:
            print(" - Closest point to robot is", point)
            print(" - Distance to robot is", distances[idx])
        # Compute distance to the center of the object
        distance = np.linalg.norm(point[:2] - center_xyz[:2].cpu().numpy())
        # Take a step towards the center of the object
        dxyz = (center_xyz - point).cpu().numpy()
        point[:2] = point[:2] + (
            dxyz[:2] / np.linalg.norm(dxyz[:2]) * min(distance, self.place_step_size)
        )
        if self.verbose:
            print(" - After taking a step towards the center of the object, we are at", point)
            print(
                " - Distance to the center of the object is",
                np.linalg.norm(point[:2] - xyt[:2]),
            )
        return point

    def can_start(self) -> bool:
        self.attempt(
            "will start placing the object if we have object and receptacle, and are close enough to drop."
        )
        if self.agent.current_object is None or self.agent.current_receptacle is None:
            self.error("Object or receptacle not found.")
            return False
        # TODO: this should be deteriministic
        # It currently is, but if you change this to something sampling-base dwe must update the test
        object_xyz = self.sample_placement_position(self.robot.get_base_pose())
        start = self.robot.get_base_pose()
        dist = np.linalg.norm(object_xyz[:2] - start[:2])
        # Check if the object is close enough to place upon
        # We need to be within the manipulation radius + place_step_size + voxel_size
        # Manipulation radius is the distance from the base to the end effector - this is what we actually plan for
        # We take a step of size place_step_size towards the object center
        # Base location sampling is often checked against voxel map - so we can have an error of up to voxel_size
        if dist > self.agent.manipulation_radius + self.place_step_size + self.agent.voxel_size:
            self.error(
                f"Object is too far away to grasp: {dist} vs {self.agent.manipulation_radius + self.place_step_size}"
            )
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
        self.robot.switch_to_manipulation_mode()

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
        if self.talk:
            self.agent.robot_say("Trying to place the object on the receptacle.")
        if not success:
            self.error("Could not place object!")
            return

        # Move to the target joint state
        self.robot.switch_to_manipulation_mode()
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

        self.agent.robot_say("I am done placing the object.")
        self.cheer("We believe we successfully placed the object.")

    def was_successful(self):
        self.error("Success detection not implemented.")
        return self._successful
