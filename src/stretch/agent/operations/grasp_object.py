# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time
import timeit
from typing import Optional, Tuple

import cv2
import numpy as np

import stretch.motion.constants as constants
from stretch.agent.base import ManagedOperation
from stretch.core.interfaces import Observations
from stretch.mapping.instance import Instance
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.filters import MaskTemporalFilter
from stretch.utils.geometry import point_global_to_base
from stretch.utils.gripper import GripperArucoDetector
from stretch.utils.point_cloud import show_point_cloud


class GraspObjectOperation(ManagedOperation):
    """Move the robot to grasp, using the end effector camera."""

    use_pitch_from_vertical: bool = True
    lift_distance: float = 0.2
    servo_to_grasp: bool = False
    _success: bool = False

    # Task information
    match_method: str = "class"
    target_object: Optional[str] = None

    # Should we use the previous mask at all?
    use_prev_mask: bool = False

    # Debugging UI elements
    show_object_to_grasp: bool = False
    show_servo_gui: bool = True
    show_point_cloud: bool = False

    # ------------------------
    # These are the most important parameters for tuning to make the grasping "feel" nice
    # This is almost certainly worth tuning a bit.

    # Thresholds for centering on object
    # align_x_threshold: int = 10
    # align_y_threshold: int = 7
    # These are the values used to decide when it's aligned enough to grasp
    align_x_threshold: int = 15
    align_y_threshold: int = 15

    # pregrasp_distance_from_object: float = 0.075
    pregrasp_distance_from_object: float = 0.25

    # This is the distance at which we close the gripper when visual servoing
    # median_distance_when_grasping: float = 0.175
    # median_distance_when_grasping: float = 0.15
    median_distance_when_grasping: float = 0.2
    lift_min_height: float = 0.1
    lift_max_height: float = 0.5

    # Movement parameters
    lift_arm_ratio: float = 0.08
    base_x_step: float = 0.10
    wrist_pitch_step: float = 0.075  # Maybe too fast
    # wrist_pitch_step: float = 0.06
    # wrist_pitch_step: float = 0.05  # Too slow
    # ------------------------

    # Parameters about how to grasp - less important
    grasp_loose: bool = False
    reset_observation: bool = False
    # Move the arm forward by this amount when grasping
    _grasp_arm_offset: float = 0.13
    # Move the arm down by this amount when grasping
    _grasp_lift_offset: float = -0.10

    # Visual servoing config
    track_image_center: bool = False
    gripper_aruco_detector: GripperArucoDetector = None
    min_points_to_approach: int = 100
    detected_center_offset_y: int = -40
    percentage_of_image_when_grasping: float = 0.2
    open_loop_z_offset: float = -0.1
    open_loop_x_offset: float = -0.1
    max_failed_attempts: int = 10
    max_random_motions: int = 10

    # Timing issues
    expected_network_delay = 0.4
    open_loop: bool = False

    # Observation memory
    observations = MaskTemporalFilter(
        observation_history_window_size_secs=5.0, observation_history_window_size_n=3
    )

    def configure(
        self,
        target_object: str,
        show_object_to_grasp: bool = False,
        servo_to_grasp: bool = False,
        show_servo_gui: bool = True,
        show_point_cloud: bool = False,
        reset_observation: bool = False,
        grasp_loose: bool = False,
    ):
        """Configure the operation with the given keyword arguments.

        Args:
            show_object_to_grasp (bool, optional): Show the object to grasp. Defaults to False.
            servo_to_grasp (bool, optional): Use visual servoing to grasp. Defaults to False.
            show_servo_gui (bool, optional): Show the servo GUI. Defaults to True.
            show_point_cloud (bool, optional): Show the point cloud. Defaults to False.
            reset_observation (bool, optional): Reset the observation. Defaults to False.
            grasp_loose (bool, optional): Grasp loosely. Useful for grasping some objects like cups. Defaults to False.
        """
        self.target_object = target_object
        self.show_object_to_grasp = show_object_to_grasp
        self.servo_to_grasp = servo_to_grasp
        self.show_servo_gui = show_servo_gui
        self.show_point_cloud = show_point_cloud
        self.reset_observation = reset_observation
        self.grasp_loose = grasp_loose

    def _debug_show_point_cloud(self, servo: Observations, current_xyz: np.ndarray) -> None:
        """Show the point cloud for debugging purposes.

        Args:
            servo (Observations): Servo observation
            current_xyz (np.ndarray): Current xyz location
        """
        # TODO: remove this, overrides existing servo state
        # servo = self.robot.get_servo_observation()
        world_xyz = servo.get_ee_xyz_in_world_frame()
        world_xyz_head = servo.get_xyz_in_world_frame()
        all_xyz = np.concatenate([world_xyz_head.reshape(-1, 3), world_xyz.reshape(-1, 3)], axis=0)
        all_rgb = np.concatenate([servo.rgb.reshape(-1, 3), servo.ee_rgb.reshape(-1, 3)], axis=0)
        show_point_cloud(all_xyz, all_rgb / 255, orig=current_xyz)

    def can_start(self):
        """Grasping can start if we have a target object picked out, and are moving to its instance, and if the robot is ready to begin manipulation."""
        if self.target_object is None:
            self.error("No target object set.")
            return False

        if not self.robot.in_manipulation_mode():
            self.robot.switch_to_manipulation_mode()

        return self.agent.current_object is not None and self.robot.in_manipulation_mode()

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

                rgb = servo.ee_rgb * (servo.instance == iid)[:, :, None].repeat(3, axis=-1)

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

    def set_target_object_class(self, target_object: str):
        """Set the target object class.

        Args:
            target_object (str): Target object class
        """
        self.target_object = target_object

    def get_target_mask(
        self,
        servo: Observations,
        instance: Instance,
        center: Tuple[int, int],
        prev_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Get target mask to move to. If we do not provide the mask from the previous step, we will simply find the mask with the most points of the correct class. Otherwise, we will try to find the mask that most overlaps with the previous mask. There are two options here: one where we simply find the mask with the most points, and another where we try to find the mask that most overlaps with the previous mask. This is in case we are losing track of particular objects and getting classes mixed up.

        Args:
            servo (Observations): Servo observation
            instance (Instance): Instance we are trying to grasp
            prev_mask (Optional[np.ndarray], optional): Mask from the previous step. Defaults to None.

        Returns:
            Optional[np.ndarray]: Target mask to move to
        """
        # Find the best masks
        class_mask = self.get_class_mask(servo)
        instance_mask = servo.instance
        if servo.ee_xyz is None:
            servo.compute_ee_xyz()

        target_mask = None
        target_mask_pts = float("-inf")
        maximum_overlap_mask = None
        maximum_overlap_pts = float("-inf")
        center_x, center_y = center

        # Loop over all detected instances -- finding the one that matches our target object
        for iid in np.unique(instance_mask):
            current_instance_mask = instance_mask == iid

            # If we are centered on the mask and it's the right class, just go for it
            if class_mask[center_y, center_x] > 0 and current_instance_mask[center_y, center_x] > 0:
                # This is the correct one - it's centered and the right class. Just go there.
                print("!!! CENTERED ON THE RIGHT OBJECT !!!")
                return current_instance_mask

            # Option 2 - try to find the map that most overlapped with what we were just trying to grasp
            # This is in case we are losing track of particular objects and getting classes mixed up
            if prev_mask is not None and self.use_prev_mask:
                # Find the mask with the most points
                mask = np.bitwise_and(current_instance_mask, prev_mask)
                mask = np.bitwise_and(mask, class_mask)
                num_pts = sum(mask.flatten())

                if num_pts > maximum_overlap_pts:
                    maximum_overlap_pts = num_pts
                    maximum_overlap_mask = mask

            # Simply find the mask with the most points
            mask = np.bitwise_and(current_instance_mask, class_mask)
            num_pts = sum(mask.flatten())
            if num_pts > target_mask_pts:
                target_mask = mask
                target_mask_pts = num_pts

        if maximum_overlap_pts > self.min_points_to_approach:
            return maximum_overlap_mask
        if target_mask is not None:
            return target_mask
        else:
            return prev_mask

    def _grasp(self) -> bool:
        """Helper function to close gripper around object."""
        self.cheer("Grasping object!")

        if not self.open_loop:
            joint_state = self.robot.get_joint_positions()
            # Now compute what to do
            base_x = joint_state[HelloStretchIdx.BASE_X]
            wrist_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
            arm = joint_state[HelloStretchIdx.ARM]
            lift = joint_state[HelloStretchIdx.LIFT]

            # Move the arm in closer
            self.robot.arm_to(
                [
                    base_x,
                    np.clip(
                        lift + self._grasp_lift_offset, self.lift_min_height, self.lift_max_height
                    ),
                    arm + self._grasp_arm_offset,
                    0,
                    wrist_pitch,
                    0,
                ],
                head=constants.look_at_ee,
                blocking=True,
            )
            time.sleep(0.5)

        self.robot.close_gripper(loose=self.grasp_loose, blocking=True)
        time.sleep(0.5)

        # Get a joint state for the object
        joint_state = self.robot.get_joint_positions()

        # Lifted joint state
        lifted_joint_state = joint_state.copy()
        lifted_joint_state[HelloStretchIdx.LIFT] += 0.2
        self.robot.arm_to(lifted_joint_state, head=constants.look_at_ee, blocking=True)
        return True

    def visual_servo_to_object(
        self, instance: Instance, max_duration: float = 120.0, max_not_moving_count: int = 50
    ) -> bool:
        """Use visual servoing to grasp the object."""

        self.intro(f"Visual servoing to grasp object {instance.global_id} {instance.category_id=}.")
        if self.show_servo_gui:
            self.warn("If you want to stop the visual servoing with the GUI up, press 'q'.")

        t0 = timeit.default_timer()
        aligned_once = False
        pregrasp_done = False
        prev_target_mask = None
        success = False
        prev_lift = float("Inf")

        # Track the fingertips using aruco markers
        if self.gripper_aruco_detector is None:
            self.gripper_aruco_detector = GripperArucoDetector()

        # Track the last object location and the number of times we've failed to grasp
        current_xyz = None
        failed_counter = 0
        not_moving_count = 0
        q_last = np.array([0.0 for _ in range(11)])  # 11 DOF, HelloStretchIdx
        random_motion_counter = 0

        if not pregrasp_done:
            # Move to pregrasp position
            self.pregrasp_open_loop(
                instance.get_center(), distance_from_object=self.pregrasp_distance_from_object
            )
            pregrasp_done = True

        # Give a short pause here to make sure ee image is up to date
        time.sleep(0.5)
        self.warn("Starting visual servoing.")

        # Main loop - run unless we time out, blocking.
        while timeit.default_timer() - t0 < max_duration:

            # Get servo observation
            servo = self.robot.get_servo_observation()
            joint_state = self.robot.get_joint_positions()
            world_xyz = servo.get_ee_xyz_in_world_frame()

            if not self.open_loop:
                # Now compute what to do
                base_x = joint_state[HelloStretchIdx.BASE_X]
                wrist_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
                arm = joint_state[HelloStretchIdx.ARM]
                lift = joint_state[HelloStretchIdx.LIFT]

            # Compute the center of the image that we will be tracking
            if self.track_image_center:
                center_x, center_y = servo.ee_rgb.shape[1] // 2, servo.ee_rgb.shape[0] // 2
            else:
                center = self.gripper_aruco_detector.detect_center(servo.ee_rgb)
                if center is not None:
                    center_y, center_x = np.round(center).astype(int)
                    center_y += self.detected_center_offset_y
                else:
                    center_x, center_y = servo.ee_rgb.shape[1] // 2, servo.ee_rgb.shape[0] // 2

            # add offset to center
            center_x -= 10  # move closer to top

            # Run semantic segmentation on it
            servo = self.agent.semantic_sensor.predict(servo, ee=True)
            latest_mask = self.get_target_mask(
                servo, instance, prev_mask=prev_target_mask, center=(center_x, center_y)
            )
            print("latest mask size:", np.sum(latest_mask.flatten()))

            # dilate mask
            kernel = np.ones((3, 3), np.uint8)
            mask_np = latest_mask.astype(np.uint8)
            dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
            latest_mask = dilated_mask.astype(bool)

            # push to history
            self.observations.push_mask_to_observation_history(
                observation=latest_mask,
                timestamp=time.time(),
                mask_size_threshold=self.min_points_to_approach,
                acquire_lock=True,
            )

            target_mask = self.observations.get_latest_observation()
            if target_mask is None:
                target_mask = np.zeros([servo.ee_rgb.shape[0], servo.ee_rgb.shape[1]], dtype=bool)

            # Get depth
            center_depth = servo.ee_depth[center_y, center_x]

            # Compute the center of the mask in image coords
            mask_center = self.observations.get_latest_centroid()
            if mask_center is None:
                # if not aligned_once:
                #     self.error(
                #         "Lost track before even seeing object with EE camera. Just try open loop."
                #     )
                #     if self.show_servo_gui:
                #         cv2.destroyAllWindows()
                # return False
                if failed_counter < self.max_failed_attempts:
                    mask_center = np.array([center_y, center_x])
                else:
                    # If we are aligned, but we lost the object, just try to grasp it
                    self.error(f"Lost track. Trying to grasp at {current_xyz}.")
                    if current_xyz is not None:
                        current_xyz[0] += self.open_loop_x_offset
                        current_xyz[2] += self.open_loop_z_offset
                    if self.show_servo_gui:
                        cv2.destroyAllWindows()
                    return self.grasp_open_loop(current_xyz)
            else:
                failed_counter = 0
                mask_center = mask_center.astype(int)
                assert (
                    world_xyz.shape[0] == servo.semantic.shape[0]
                    and world_xyz.shape[1] == servo.semantic.shape[1]
                ), "World xyz shape does not match semantic shape."
                current_xyz = world_xyz[int(mask_center[0]), int(mask_center[1])]
                if self.show_point_cloud:
                    self._debug_show_point_cloud(servo, current_xyz)

            # Optionally display which object we are servoing to
            if self.show_servo_gui:
                servo_ee_rgb = cv2.cvtColor(servo.ee_rgb, cv2.COLOR_RGB2BGR)
                mask = target_mask.astype(np.uint8) * 255
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask[:, :, 0] = 0
                servo_ee_rgb = cv2.addWeighted(servo_ee_rgb, 0.5, mask, 0.5, 0, servo_ee_rgb)
                # Draw the center of the image
                servo_ee_rgb = cv2.circle(servo_ee_rgb, (center_x, center_y), 5, (255, 0, 0), -1)
                # Draw the center of the mask
                servo_ee_rgb = cv2.circle(
                    servo_ee_rgb, (int(mask_center[1]), int(mask_center[0])), 5, (0, 255, 0), -1
                )
                cv2.imshow("servo_ee_rgb", servo_ee_rgb)
                cv2.waitKey(1)
                res = cv2.waitKey(1) & 0xFF  # 0xFF is a mask to get the last 8 bits
                if res == ord("q"):
                    break

            # check not moving threshold
            if not_moving_count > max_not_moving_count:
                self.info("Not moving; try to grasp.")
                success = self._grasp()
                break
            # If we have a target mask, compute the median depth of the object
            # Otherwise we will just try to grasp if we are close enough - assume we lost track!
            if target_mask is not None:
                object_depth = servo.ee_depth[target_mask]
                median_object_depth = np.median(servo.ee_depth[target_mask]) / 1000
            else:
                print("detected classes:", np.unique(servo.ee_semantic))
                if center_depth < self.median_distance_when_grasping:
                    success = self._grasp()
                continue

            dx, dy = mask_center[1] - center_x, mask_center[0] - center_y

            # Is the center of the image part of the target mask or not?
            center_in_mask = target_mask[int(center_y), int(center_x)] > 0
            # TODO: add deadband bubble around this?

            # Since we were able to detect it, copy over the target mask
            prev_target_mask = target_mask

            # Are we aligned to the object
            aligned = np.abs(dx) < self.align_x_threshold and np.abs(dy) < self.align_y_threshold

            print()
            print("----- STEP VISUAL SERVOING -----")
            print("Observed this many target mask points:", np.sum(target_mask.flatten()))
            print("failed =", failed_counter, "/", self.max_failed_attempts)
            print("cur x =", base_x)
            print(" lift =", lift)
            print("  arm =", arm)
            print("pitch =", wrist_pitch)
            print("Center depth:", center_depth)
            print(f"base_x={base_x}, wrist_pitch={wrist_pitch}, dx={dx}, dy={dy}")
            print(f"Median distance to object is {median_object_depth}.")
            print(f"Center distance to object is {center_depth}.")
            print("Center in mask?", center_in_mask)
            print("Current XYZ:", current_xyz)
            print("Aligned?", aligned)

            # Fix lift to only go down
            lift = min(lift, prev_lift)

            if aligned:
                # First, check to see if we are close enough to grasp
                if center_depth < self.median_distance_when_grasping:
                    print(
                        f"Center depth of {center_depth} is close enough to grasp; less than {self.median_distance_when_grasping}."
                    )
                    self.info("Aligned and close enough to grasp.")
                    success = self._grasp()
                    break

                # If we are aligned, step the whole thing closer by some amount
                # This is based on the pitch - basically
                aligned_once = True
                arm_component = np.cos(wrist_pitch) * self.lift_arm_ratio
                lift_component = np.sin(wrist_pitch) * self.lift_arm_ratio
                arm += arm_component
                lift += lift_component
            else:
                # Add these to do some really hacky proportionate control
                px = max(0.25, np.abs(2 * dx / target_mask.shape[1]))
                py = max(0.25, np.abs(2 * dy / target_mask.shape[0]))

                # Move the base and modify the wrist pitch
                # TODO: remove debug code
                # print(f"dx={dx}, dy={dy}, px={px}, py={py}")
                if dx > self.align_x_threshold:
                    # Move in x - this means translate the base
                    base_x += -self.base_x_step * px
                elif dx < -1 * self.align_x_threshold:
                    base_x += self.base_x_step * px
                if dy > self.align_y_threshold:
                    # Move in y - this means translate the base
                    wrist_pitch += -self.wrist_pitch_step * py
                elif dy < -1 * self.align_y_threshold:
                    wrist_pitch += self.wrist_pitch_step * py

            # Force to reacquire the target mask if we moved the camera too much
            prev_target_mask = None

            # safety checks
            q = [
                base_x,
                0.0,
                0.0,
                lift,
                arm,
                0.0,
                0.0,
                wrist_pitch,
                0.0,
                0.0,
                0.0,
            ]  # 11 DOF: see HelloStretchIdx

            q = np.array(q)

            ee_pos, ee_quat = self.robot_model.manip_fk(q)

            while ee_pos[2] < 0.03:
                lift += 0.01
                q[HelloStretchIdx.LIFT] = lift
                ee_pos, ee_quat = self.robot_model.manip_fk(q)

            print("tgt x =", base_x)
            print(" lift =", lift)
            print("  arm =", arm)
            print("pitch =", wrist_pitch)

            self.robot.arm_to(
                [base_x, lift, arm, 0, wrist_pitch, 0],
                head=constants.look_at_ee,
                blocking=False,
            )
            prev_lift = lift
            time.sleep(self.expected_network_delay)

            # check not moving
            if np.linalg.norm(q - q_last) < 0.05:  # TODO: tune
                not_moving_count += 1
            else:
                not_moving_count = 0

            q_last = q

            if random_motion_counter > self.max_random_motions:
                self.error("Failed to align to object after 10 random motions.")
                break

        if self.show_servo_gui:
            cv2.destroyAllWindows()
        return success

    def run(self) -> None:
        self.intro("Grasping the object.")
        self._success = False
        if self.show_object_to_grasp:
            self.show_instance(self.agent.current_object)

        assert self.target_object is not None, "Target object must be set before running."

        # Now we should be able to see the object if we orient gripper properly
        # Get the end effector pose
        obs = self.robot.get_observation()
        joint_state = self.robot.get_joint_positions()
        model = self.robot.get_robot_model()

        if joint_state[HelloStretchIdx.GRIPPER] < 0.0:
            self.robot.open_gripper(blocking=True)

        # Get the current base pose of the robot
        xyt = self.robot.get_base_pose()

        # Note that these are in the robot's current coordinate frame; they're not global coordinates, so this is ok to use to compute motions.
        object_xyz = self.agent.current_object.get_center()
        relative_object_xyz = point_global_to_base(object_xyz, xyt)

        # Compute the angles necessary
        if self.use_pitch_from_vertical:
            ee_pos, ee_rot = model.manip_fk(joint_state)
            dy = np.abs(ee_pos[1] - relative_object_xyz[1])
            dz = np.abs(ee_pos[2] - relative_object_xyz[2])
            pitch_from_vertical = np.arctan2(dy, dz)
        else:
            pitch_from_vertical = 0.0

        # Compute final pregrasp joint state goal and send the robot there
        joint_state[HelloStretchIdx.WRIST_PITCH] = -np.pi / 2 + pitch_from_vertical
        self.robot.arm_to(joint_state, head=constants.look_at_ee, blocking=True)

        if self.servo_to_grasp:
            # If we try to servo, then do this
            self._success = self.visual_servo_to_object(self.agent.current_object)

        # clear observations
        if self.reset_observation:
            self.observations.clear_history()
            self.agent.reset_object_plans()
            self.agent.voxel_map.instances.pop_global_instance(
                env_id=0, global_instance_id=self.agent.current_object.global_id
            )

    def pregrasp_open_loop(self, object_xyz: np.ndarray, distance_from_object: float = 0.25):
        """Move to a pregrasp position in an open loop manner.

        Args:
            object_xyz (np.ndarray): Location to grasp
            distance_from_object (float, optional): Distance from object. Defaults to 0.2.
        """
        xyt = self.robot.get_base_pose()
        relative_object_xyz = point_global_to_base(object_xyz, xyt)

        joint_state = self.robot.get_joint_positions()

        model = self.robot.get_robot_model()
        ee_pos, ee_rot = model.manip_fk(joint_state)

        vector_to_object = relative_object_xyz - ee_pos
        vector_to_object = vector_to_object / np.linalg.norm(vector_to_object)

        shifted_object_xyz = relative_object_xyz - (distance_from_object * vector_to_object)

        # IK
        target_joint_positions, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            shifted_object_xyz, ee_rot, q0=joint_state
        )
        print("Pregrasp joint positions: ")
        print(target_joint_positions)

        # get point 10cm from object
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

        # Move to the target joint state
        robot_pose = [
            target_joint_positions[HelloStretchIdx.BASE_X],
            target_joint_positions[HelloStretchIdx.LIFT],
            target_joint_positions[HelloStretchIdx.ARM],
            0.0,
            target_joint_positions[HelloStretchIdx.WRIST_PITCH],
            0.0,
        ]
        print(f"{self.name}: Moving to pre-grasp position.")
        self.robot.arm_to(target_joint_positions, head=constants.look_at_ee, blocking=True)

        # wait for image to stabilize
        time.sleep(1.0)

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

    def was_successful(self) -> bool:
        """Return true if successful"""
        return self._success
