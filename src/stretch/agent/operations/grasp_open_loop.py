# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time

import numpy as np

import stretch.motion.constants as constants
from stretch.agent.base import ManagedOperation
from stretch.core.interfaces import Observations

# from stretch.mapping.instance import Instance
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.geometry import point_global_to_base
from stretch.utils.point_cloud import show_point_cloud


class OpenLoopGraspObjectOperation(ManagedOperation):

    debug_show_point_cloud: bool = False
    match_method: str = "feature"
    target_object: str = None
    lift_distance: float = 0.2

    def configure(
        self,
        target_object: str,
        match_method: str = "feature",
        debug_show_point_cloud: bool = False,
        lift_distance: float = 0.2,
    ) -> None:
        self.target_object = target_object
        self.match_method = match_method
        self.debug_show_point_cloud = debug_show_point_cloud
        self.lift_distance = lift_distance

    def set_target_object_class(self, target_object: str) -> None:
        self.target_object = target_object

    def set_target_object(self, target_object: str) -> None:
        self.target_object = target_object

    def can_start(self):
        """Grasping can start if we have a target object picked out, and are moving to its instance, and if the robot is ready to begin manipulation."""
        if self.target_object is None:
            self.error("No target object set.")
            return False
        return self.agent.current_object is not None and self.robot.in_manipulation_mode()

    def was_successful(self) -> bool:
        """Return true if successful"""
        return self._success

    def _debug_show_point_cloud(self, servo: Observations, current_xyz: np.ndarray) -> None:
        """Show the point cloud for debugging purposes.

        Args:
            servo (Observations): Servo observation
            current_xyz (np.ndarray): Current xyz location
        """
        world_xyz_head = servo.get_xyz_in_world_frame()
        all_xyz = world_xyz_head.reshape(-1, 3)
        all_rgb = servo.rgb.reshape(-1, 3) / 255
        show_point_cloud(all_xyz, all_rgb, orig=current_xyz)

    def get_class_mask(self, servo: Observations) -> np.ndarray:
        """Get the mask for the class of the object we are trying to grasp. Multiple options might be acceptable.

        Args:
            servo (Observations): Servo observation

        Returns:
            np.ndarray: Mask for the class of the object we are trying to grasp
        """
        mask = np.zeros_like(servo.semantic).astype(bool)  # type: ignore

        print("[GRASP OBJECT] match method =", self.match_method)
        if self.match_method == "class":

            # Get the target class
            if self.agent.current_object is not None:
                target_class_id = self.agent.current_object.category_id
                target_class = self.agent.semantic_sensor.get_class_name_for_id(target_class_id)
            else:
                target_class = self.target_object

            print("[GRASP OBJECT] Detecting objects of class", target_class)

            # Now find the mask with that class
            for iid in np.unique(servo.semantic):
                name = self.agent.semantic_sensor.get_class_name_for_id(iid)
                if name is not None and target_class in name:
                    mask = np.bitwise_or(mask, servo.semantic == iid)
        elif self.match_method == "feature":
            if self.target_object is None:
                raise ValueError(
                    f"Target object must be set before running match method {self.match_method}."
                )
            print("[GRASP OBJECT] Detecting objects described as", self.target_object)
            text_features = self.agent.encode_text(self.target_object)
            best_score = float("-inf")
            best_iid = None

            for iid in np.unique(servo.instance):

                # Ignore the background
                if iid < 0:
                    continue

                rgb = servo.rgb * (servo.instance == iid)[:, :, None].repeat(3, axis=-1)

                # TODO: remove debug code
                #  import matplotlib.pyplot as plt
                # plt.imshow(rgb)
                # plt.show()

                features = self.agent.encode_image(rgb)
                score = self.agent.compare_features(text_features, features)
                print(f" - Score for {iid} is {score}")
                if score > best_score:
                    best_score = score
                    best_iid = iid
                    mask = servo.instance == iid
        else:
            raise ValueError(f"Invalid matching method {self.match_method}.")

        return mask

    def run(self):
        """Grasp based on data from the head camera only."""
        self.attempt("Grasping an object")

        # Get the arm out of the way so we can look for the object
        q = self.robot.get_joint_positions()
        base_x = q[HelloStretchIdx.BASE_X]
        lift = q[HelloStretchIdx.LIFT]
        arm = q[HelloStretchIdx.ARM]
        wrist_pitch = q[HelloStretchIdx.WRIST_PITCH]
        wrist_yaw = q[HelloStretchIdx.WRIST_YAW]
        wrist_roll = q[HelloStretchIdx.WRIST_ROLL]
        self.robot.arm_to(
            [base_x, 0.3, arm, wrist_roll, wrist_pitch, wrist_yaw],
            head=[-np.pi / 2, -3 * np.pi / 8],
            blocking=True,
        )
        time.sleep(0.25)

        # Capture an observation
        obs = self.robot.get_observation()
        obs = self.agent.semantic_sensor.predict(obs, ee=False)
        current_xyz = obs.get_xyz_in_world_frame()

        # Find the best object mask
        mask = self.get_class_mask(obs)

        # If the mask is empty, just use blind grasp based on instance center
        if np.sum(mask) == 0:
            self.warn("No mask found, using blind grasp.")
            object_xyz = self.agent.current_object.get_center()
        else:
            # Find the object location
            object_points = current_xyz[mask, :]
            object_xyz = np.mean(object_points, axis=0)
            self.info(f"Object location: {object_xyz}")

        if self.debug_show_point_cloud:
            self._debug_show_point_cloud(obs, object_xyz)

        # Grasp the object
        self.grasp_open_loop(object_xyz)

    def grasp_open_loop(self, object_xyz: np.ndarray):
        """Grasp the object in an open loop manner. We will just move to object_xyz and close the gripper.

        Args:
            object_xyz (np.ndarray): Location to grasp

        Returns:
            bool: True if successful, False otherwise
        """

        model = self.robot.get_robot_model()
        xyt = self.robot.get_base_pose()
        relative_object_xyz = point_global_to_base(object_xyz, xyt)

        joint_state = self.robot.get_joint_positions()

        # We assume the current end-effector orientation is the correct one, going into this
        ee_pos, ee_rot = model.manip_fk(joint_state)

        # If we failed, or if we are not servoing, then just move to the object
        target_joint_positions, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            relative_object_xyz, ee_rot, q0=joint_state
        )
        target_joint_positions[HelloStretchIdx.BASE_X] -= 0.04
        if not success:
            print("Failed to find a valid IK solution.")
            self._success = False
            return
        elif (
            target_joint_positions[HelloStretchIdx.ARM] < 0
            or target_joint_positions[HelloStretchIdx.LIFT] < 0
        ):
            print(
                f"{self.name}: Target joint state is invalid: {target_joint_positions}. Positions for arm and lift must be positive."
            )
            self._success = False
            return

        # Lift the arm up a bit
        target_joint_positions_lifted = target_joint_positions.copy()
        target_joint_positions_lifted[HelloStretchIdx.LIFT] += self.lift_distance

        print("Go to lifted target position")
        self.robot.arm_to(target_joint_positions_lifted, head=constants.look_at_ee, blocking=True)
        time.sleep(0.5)

        # Move to the target joint state
        print(f"{self.name}: Moving to grasp position.")
        self.robot.arm_to(target_joint_positions, head=constants.look_at_ee, blocking=True)
        time.sleep(0.5)
        print(f"{self.name}: Closing the gripper.")
        self.robot.close_gripper(blocking=True)
        time.sleep(0.5)
        print(f"{self.name}: Lifting the arm up so as not to hit the base.")
        self.robot.arm_to(target_joint_positions_lifted, head=constants.look_at_ee, blocking=False)
        print(f"{self.name}: Return arm to initial configuration.")
        self.robot.arm_to(joint_state, head=constants.look_at_ee, blocking=True)
        print(f"{self.name}: Done.")
        self._success = True
        return
