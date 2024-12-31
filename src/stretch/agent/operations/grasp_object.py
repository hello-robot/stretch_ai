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

import os
import time
import timeit
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

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
    talk: bool = True
    verbose: bool = False

    offset_from_vertical = -np.pi / 2 - 0.1

    # Task information
    match_method: str = "class"
    target_object: Optional[str] = None
    _object_xyz: Optional[np.ndarray] = None

    # Should we use the previous mask at all?
    use_prev_mask: bool = False

    # Debugging UI elements
    show_object_to_grasp: bool = False
    show_servo_gui: bool = False
    show_point_cloud: bool = False
    debug_grasping: bool = False

    # This will delete the object from instance memory/voxel map after grasping
    delete_object_after_grasp: bool = False

    # Should we try grasping open-loop or not?
    _try_open_loop: bool = False

    # ------------------------
    # These are the most important parameters for tuning to make the grasping "feel" nice
    # This is almost certainly worth tuning a bit.

    # Thresholds for centering on object
    # These are the values used to decide when it's aligned enough to grasp
    align_x_threshold: int = 30
    align_y_threshold: int = 25

    # This is the distance before we start servoing to the object
    # Standoff distance from actual grasp pose
    pregrasp_distance_from_object: float = 0.3

    # ------------------------
    # Grasping motion planning parameters and offsets
    # This is the distance at which we close the gripper when visual servoing
    median_distance_when_grasping: float = 0.18
    lift_min_height: float = 0.1
    lift_max_height: float = 1.0

    # How long is the gripper?
    # This is used to compute when we should not move the robot forward any farther
    # grasp_distance = 0.12
    grasp_distance = 0.14

    # Movement parameters
    lift_arm_ratio: float = 0.05
    base_x_step: float = 0.10
    wrist_pitch_step: float = 0.2  # 075  # Maybe too fast
    # ------------------------

    # Tracked object features for making sure we are grabbing the right thing
    tracked_object_features: Optional[torch.Tensor] = None

    # Parameters about how to grasp - less important
    grasp_loose: bool = False
    reset_observation: bool = False
    # Move the arm forward by this amount when grasping
    _grasp_arm_offset: float = 0.0  # 0.13
    # Move the arm down by this amount when grasping
    _grasp_lift_offset: float = 0.0  # -0.05

    # Visual servoing config
    track_image_center: bool = False
    gripper_aruco_detector: GripperArucoDetector = None
    min_points_to_approach: int = 100
    detected_center_offset_x: int = 0  # -10
    detected_center_offset_y: int = 0  # -40
    percentage_of_image_when_grasping: float = 0.2
    open_loop_z_offset: float = -0.1
    open_loop_x_offset: float = -0.1
    max_failed_attempts: int = 10
    max_random_motions: int = 10

    # Timing issues
    expected_network_delay = 0.1
    open_loop: bool = False

    # Observation memory
    observations = MaskTemporalFilter(
        observation_history_window_size_secs=5.0, observation_history_window_size_n=3
    )

    def configure(
        self,
        target_object: Optional[str] = None,
        object_xyz: Optional[np.ndarray] = None,
        show_object_to_grasp: bool = False,
        servo_to_grasp: bool = True,
        show_servo_gui: bool = True,
        show_point_cloud: bool = False,
        reset_observation: bool = False,
        grasp_loose: bool = False,
        talk: bool = True,
        match_method: str = "class",
        delete_object_after_grasp: bool = True,
        try_open_loop: bool = False,
    ):
        """Configure the operation with the given keyword arguments.

        Args:
            show_object_to_grasp (bool, optional): Show the object to grasp. Defaults to False.
            servo_to_grasp (bool, optional): Use visual servoing to grasp. Defaults to False.
            show_servo_gui (bool, optional): Show the servo GUI. Defaults to True.
            show_point_cloud (bool, optional): Show the point cloud. Defaults to False.
            reset_observation (bool, optional): Reset the observation. Defaults to False.
            grasp_loose (bool, optional): Grasp loosely. Useful for grasping some objects like cups. Defaults to False.
            talk (bool, optional): Talk as the robot tries to grab stuff. Defaults to True.
            match_method (str, optional): Matching method. Defaults to "class". This is how the policy determines which object mask it should try to grasp.
            delete_object_after_grasp (bool, optional): Delete the object after grasping. Defaults to True.
            try_open_loop (bool, optional): Try open loop grasping. Defaults to False.
        """
        if target_object is not None:
            self.target_object = target_object
        if object_xyz is not None:
            assert len(object_xyz) == 3, "Object xyz must be a 3D point."
            self._object_xyz = object_xyz
        self.show_object_to_grasp = show_object_to_grasp
        self.servo_to_grasp = servo_to_grasp
        self.show_servo_gui = show_servo_gui
        self.show_point_cloud = show_point_cloud
        self.reset_observation = reset_observation
        self.delete_object_after_grasp = delete_object_after_grasp
        self.grasp_loose = grasp_loose
        self.talk = talk
        self.match_method = match_method
        self._try_open_loop = try_open_loop
        if self.match_method not in ["class", "feature"]:
            raise ValueError(
                f"Unknown match method {self.match_method}. Should be 'class' or 'feature'."
            )

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

        return (
            self.agent.current_object is not None or self._object_xyz is not None
        ) and self.robot.in_manipulation_mode()

    def _compute_center_depth(
        self,
        servo: Observations,
        target_mask: np.ndarray,
        center_y: int,
        center_x: int,
        local_region_size: int = 5,
    ) -> float:
        """Compute the center depth of the object.

        Args:
            servo (Observations): Servo observation
            target_mask (np.ndarray): Target mask
            center_y (int): Center y
            center_x (int): Center x
            local_region_size (int, optional): Local region size. Defaults to 5.

        Returns:
            float: Center depth of the object
        """
        # Compute depth as median of object pixels near center_y, center_x
        # Make a mask of radius 10 around the center
        mask = np.zeros_like(target_mask)
        mask[
            max(center_y - local_region_size, 0) : min(center_y + local_region_size, mask.shape[0]),
            max(center_x - local_region_size, 0) : min(center_x + local_region_size, mask.shape[1]),
        ] = 1

        # Ignore depth of 0 (bad value)
        depth_mask = np.bitwise_and(servo.ee_depth > 1e-8, mask)

        depth = servo.ee_depth[target_mask & depth_mask]
        if len(depth) == 0:
            return 0.0

        median_depth = np.median(depth)

        return median_depth

    def get_class_mask(self, servo: Observations) -> np.ndarray:
        """Get the mask for the class of the object we are trying to grasp. Multiple options might be acceptable.

        Args:
            servo (Observations): Servo observation

        Returns:
            np.ndarray: Mask for the class of the object we are trying to grasp
        """
        mask = np.zeros_like(servo.semantic).astype(bool)  # type: ignore

        if self.verbose:
            print("[GRASP OBJECT] match method =", self.match_method)
        if self.match_method == "class":

            # Get the target class
            if self.agent.current_object is not None:
                target_class_id = self.agent.current_object.category_id
                target_class = self.agent.semantic_sensor.get_class_name_for_id(target_class_id)
            else:
                target_class = self.target_object

            if self.verbose:
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

            if self.verbose:
                print("[GRASP OBJECT] Detecting objects described as", self.target_object)

            text_features = self.agent.encode_text(self.target_object)
            best_score = float("-inf")
            best_iid = None
            all_matches = []

            # Loop over all detected instances
            for iid in np.unique(servo.instance):

                # Ignore the background
                if iid < 0:
                    continue

                rgb = servo.ee_rgb * (servo.instance == iid)[:, :, None].repeat(3, axis=-1)

                features = self.agent.encode_image(rgb)
                score = self.agent.compare_features(text_features, features).item()

                # if self.verbose:
                print(
                    f" - Score for {iid} is {score} / {self.agent.grasp_feature_match_threshold}."
                )

                # Score is determined based on the feature comparison
                if score > best_score:
                    best_score = score
                    best_iid = iid
                if score > self.agent.feature_match_threshold:
                    all_matches.append((score, iid, features))
            if len(all_matches) > 0:
                print("[MASK SELECTION] All matches:")
                for score, iid, features in all_matches:
                    print(f" - Matched {iid} with score {score}.")
            if len(all_matches) == 0:
                print("[MASK SELECTION] No matches found.")
            elif len(all_matches) == 1:
                print("[MASK SELECTION] One match found. We are done.")
                mask = servo.instance == best_iid
                # Set the tracked features
                self.tracked_object_features = all_matches[0][2]
            else:
                # Check to see if we have tracked features
                if self.tracked_object_features is not None:
                    # Find the closest match
                    best_score = float("-inf")
                    best_iid = None
                    best_features = None
                    for _, iid, features in all_matches:
                        score = self.agent.compare_features(self.tracked_object_features, features)
                        if score > best_score:
                            best_score = score
                            best_iid = iid
                            best_features = features
                    self.tracked_object_features = best_features
                else:
                    best_score = float("-inf")
                    best_iid = None
                    for score, iid, _ in all_matches:
                        if score > best_score:
                            best_score = score
                            best_iid = iid

                # Set the mask
                mask = servo.instance == best_iid
        else:
            raise ValueError(f"Invalid matching method {self.match_method}.")

        return mask

    def set_target_object_class(self, target_object: str):
        """Set the target object class.

        Args:
            target_object (str): Target object class
        """
        self.target_object = target_object

    def reset(self):
        """Reset the operation. This clears the history and sets the success flag to False. It also clears the tracked object features."""
        self._success = False
        self.tracked_object_features = None
        self.observations.clear_history()

    def get_target_mask(
        self,
        servo: Observations,
        center: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Get target mask to move to. If we do not provide the mask from the previous step, we will simply find the mask with the most points of the correct class. Otherwise, we will try to find the mask that most overlaps with the previous mask. There are two options here: one where we simply find the mask with the most points, and another where we try to find the mask that most overlaps with the previous mask. This is in case we are losing track of particular objects and getting classes mixed up.

        Args:
            servo (Observations): Servo observation
            center (Tuple[int, int]): Center of the image

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
            return None

    def sayable_target_object(self) -> str:
        """Get the target object in a sayable format.

        Returns:
            str: Sayable target object
        """

        # Replace underscores, etc
        return self.target_object.replace("_", " ")

    def _grasp(self, distance: Optional[float] = None) -> bool:
        """Helper function to close gripper around object.

        Returns:
            bool: True if successful, False otherwise
        """

        self.cheer("Grasping object!")
        if self.talk:
            self.agent.robot_say(f"Grasping the {self.sayable_target_object()}!")

        print("Distance:", distance)
        joint_state = self.robot.get_joint_positions()
        if not self.open_loop or distance is not None:
            # Now compute what to do
            base_x = joint_state[HelloStretchIdx.BASE_X]
            wrist_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
            arm = joint_state[HelloStretchIdx.ARM]
            lift = joint_state[HelloStretchIdx.LIFT]

            if distance is not None:
                distance = max(distance - self.grasp_distance, 0)
                print("Distance to move:", distance)
                if distance > 0:
                    # Use wrist pitch to compute arm and lift offsets
                    arm_component = np.cos(wrist_pitch) * distance
                    lift_component = np.sin(wrist_pitch) * distance
                else:
                    arm_component = 0
                    lift_component = 0
            else:
                arm_component = 0
                lift_component = 0

            # Move the arm in closer
            self.robot.arm_to(
                [
                    base_x,
                    np.clip(
                        lift + lift_component,
                        min(joint_state[HelloStretchIdx.LIFT], self.lift_min_height),
                        self.lift_max_height,
                    ),
                    arm + arm_component,
                    0,
                    wrist_pitch,
                    0,
                ],
                head=constants.look_at_ee,
                blocking=True,
            )
            time.sleep(0.1)

        self.robot.close_gripper(loose=self.grasp_loose, blocking=True)
        time.sleep(0.1)

        # Get a joint state for the object
        joint_state = self.robot.get_joint_positions()

        # Lifted joint state
        lifted_joint_state = joint_state.copy()
        lifted_joint_state[HelloStretchIdx.LIFT] += 0.2
        self.robot.arm_to(lifted_joint_state, head=constants.look_at_ee, blocking=True)
        return True

    def blue_highlight_mask(self, img):
        """Get a binary mask for the blue highlights in the image."""

        # Conditions for each channel
        blue_condition = img[:, :, 2] > 100
        red_condition = img[:, :, 0] < 50
        green_condition = img[:, :, 1] < 50

        # Combine conditions to create a binary mask
        mask = blue_condition & red_condition & green_condition

        # Convert boolean mask to binary (0 and 1)
        return mask.astype(np.uint8)

    def visual_servo_to_object(
        self, instance: Instance, max_duration: float = 120.0, max_not_moving_count: int = 50
    ) -> bool:
        """Use visual servoing to grasp the object."""

        if instance is not None:
            self.intro(
                f"Visual servoing to grasp object {instance.global_id} {instance.category_id=}."
            )
        else:
            self.intro("Visual servoing to grasp {self.target_object} at {self._object_xyz}.")

        if self.show_servo_gui:
            self.warn("If you want to stop the visual servoing with the GUI up, press 'q'.")

        t0 = timeit.default_timer()
        aligned_once = False
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

        # Track the depth of the object
        center_depth = None
        prev_center_depth = None

        # Move to pregrasp position
        self.pregrasp_open_loop(
            self.get_object_xyz(), distance_from_object=self.pregrasp_distance_from_object
        )

        # Give a short pause here to make sure ee image is up to date
        time.sleep(0.25)
        self.warn("Starting visual servoing.")

        if self.debug_grasping:
            # make debug dir
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            debug_dir_name = f"debug/debug_{current_time}"
            os.mkdir(debug_dir_name)

        # This is a counter used for saving debug images
        # It's not used in the actual grasping code
        iter_ = 0

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
            center_x += self.detected_center_offset_x  # move closer to top

            # Run semantic segmentation on it
            servo = self.agent.semantic_sensor.predict(servo, ee=True)
            latest_mask = self.get_target_mask(servo, center=(center_x, center_y))

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
            if center_depth is not None and center_depth > 1e-8:
                prev_center_depth = center_depth

            # Compute depth as median of object pixels near center_y, center_x
            center_depth = self._compute_center_depth(servo, target_mask, center_y, center_x)

            if self.debug_grasping:
                # save data
                mask = target_mask.astype(np.uint8) * 255
                debug_viz = np.zeros((240, 640, 3))
                debug_viz[:, :320, :] = servo.ee_rgb
                debug_viz[:, 320:, 0] = mask
                debug_viz[:, 320:, 1] = mask
                debug_viz[:, 320:, 2] = mask
                Image.fromarray(debug_viz.astype("uint8")).save(
                    f"{debug_dir_name}/img_{iter_:03d}.png"
                )
            iter_ += 1

            # Compute the center of the mask in image coords
            mask_center = self.observations.get_latest_centroid()
            if mask_center is None:
                failed_counter += 1
                if failed_counter < self.max_failed_attempts:
                    mask_center = np.array([center_y, center_x])
                else:
                    # If we are aligned, but we lost the object, just try to grasp it
                    self.error(f"Lost track. Trying to grasp at {current_xyz}.")
                    if current_xyz is not None:
                        current_xyz[0] += self.open_loop_x_offset
                        current_xyz[2] += self.open_loop_z_offset
                    if self.show_servo_gui and not self.headless_machine:
                        cv2.destroyAllWindows()
                    if self._try_open_loop:
                        return self.grasp_open_loop(current_xyz)
                    else:
                        if self.talk:
                            self.agent.robot_say(f"I can't see the {self.target_object}.")
                        self._success = False
                        return False
                continue
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
            if self.show_servo_gui and not self.headless_machine:
                print(" -> Displaying visual servoing GUI.")
                servo_ee_rgb = cv2.cvtColor(servo.ee_rgb, cv2.COLOR_RGB2BGR)
                mask = target_mask.astype(np.uint8) * 255
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask[:, :, 0] = 0

                # Create an RGB image with the mask overlaid
                servo_ee_rgb = cv2.addWeighted(servo_ee_rgb, 0.5, mask, 0.5, 0, servo_ee_rgb)
                # Draw the center of the image
                servo_ee_rgb = cv2.circle(servo_ee_rgb, (center_x, center_y), 5, (255, 0, 0), -1)
                # Draw the center of the mask
                servo_ee_rgb = cv2.circle(
                    servo_ee_rgb, (int(mask_center[1]), int(mask_center[0])), 5, (0, 255, 0), -1
                )

                # Create a depth image with the center of the mask
                # First convert to 32-bit float
                viz_ee_depth = cv2.normalize(servo.ee_depth, None, 0, 255, cv2.NORM_MINMAX)
                viz_ee_depth = viz_ee_depth.astype(np.uint8)
                viz_ee_depth = cv2.applyColorMap(viz_ee_depth, cv2.COLORMAP_JET)
                viz_ee_depth = cv2.circle(
                    viz_ee_depth, (int(mask_center[1]), int(mask_center[0])), 5, (0, 255, 0), -1
                )

                # Concatenate the two images side by side
                viz_image = np.concatenate([servo_ee_rgb, viz_ee_depth], axis=1)
                cv2.namedWindow("Visual Servoing", cv2.WINDOW_NORMAL)
                cv2.imshow("Visual Servoing", viz_image)
                cv2.waitKey(1)
                res = cv2.waitKey(1) & 0xFF  # 0xFF is a mask to get the last 8 bits
                if res == ord("q"):
                    break

            if self.debug_grasping:
                # show all four images
                concatenated_image = np.concatenate((debug_viz.astype("uint8"), viz_image), axis=1)
                Image.fromarray(concatenated_image).save(
                    f"{debug_dir_name}/img_point_{iter_:03d}.png"
                )

            # check not moving threshold
            if not_moving_count > max_not_moving_count:
                self.info("Not moving; try to grasp.")
                success = self._grasp()
                break

            # If we have a target mask, compute the median depth of the object
            # Otherwise we will just try to grasp if we are close enough - assume we lost track!
            if target_mask is not None:
                object_depth = servo.ee_depth[target_mask]
                median_object_depth = np.median(servo.ee_depth[target_mask])  # / 1000
            else:
                # print("detected classes:", np.unique(servo.ee_semantic))
                if center_depth < self.median_distance_when_grasping and center_depth > 1e-8:
                    success = self._grasp(distance=center_depth)
                else:
                    # Could not find the object
                    failed_counter += 1
                continue

            dx, dy = mask_center[1] - center_x, mask_center[0] - center_y

            # Is the center of the image part of the target mask or not?

            # Erode the target mask to make sure we are not just on the edge
            kernel = np.ones((4, 4), np.uint8)
            eroded_target_mask = cv2.erode(target_mask.astype(np.uint8), kernel, iterations=10)

            center_in_mask = eroded_target_mask[int(center_y), int(center_x)] > 0
            # TODO: add deadband bubble around this?

            # Are we aligned to the object
            aligned = np.abs(dx) < self.align_x_threshold and np.abs(dy) < self.align_y_threshold

            print()
            print("----- STEP VISUAL SERVOING -----")
            print("Observed this many target mask points:", np.sum(target_mask.flatten()))
            if self.verbose:
                print("failed =", failed_counter, "/", self.max_failed_attempts)
                print("cur x =", base_x)
                print(" lift =", lift)
                print("  arm =", arm)
                print("pitch =", wrist_pitch)
                print("Center depth:", center_depth, "prev :", prev_center_depth)
                print(f"base_x={base_x}, wrist_pitch={wrist_pitch}, dx={dx}, dy={dy}")
                print(f"Median distance to object is {median_object_depth}.")
            print(f"Center distance to object is {center_depth}.")
            print("Center in mask?", center_in_mask)
            print("Aligned?", aligned)

            # Fix lift to only go down
            lift = min(lift, prev_lift)

            # If we are aligned, try to grasp
            if aligned or center_in_mask:
                # First, check to see if we are close enough to grasp
                if center_depth < self.median_distance_when_grasping and center_depth > 1e-8:
                    print(
                        f"Center depth of {center_depth} is close enough to grasp; less than {self.median_distance_when_grasping}."
                    )
                    if center_depth <= 1e-8:
                        self.warn("Bad depth value; trying to grasp.")
                        self.info("Previous good depth value: " + str(prev_center_depth))
                        self.info("Perform an open-loop motion towards the object.")
                        success = self._grasp(distance=prev_center_depth)
                    else:
                        self.info("Aligned and close enough to grasp.")
                        success = self._grasp()

                    # Added debugging code to make sure that we are seeing the right object
                    if self.debug_grasping:
                        # record image
                        servo = self.robot.get_servo_observation()
                        debug_viz = np.zeros((240, 640, 3))
                        debug_viz[:, :320, :] = servo.ee_rgb
                        debug_viz[:, 320:, :] = servo.rgb
                        Image.fromarray(debug_viz.astype("uint8")).save(
                            f"{debug_dir_name}/img_point_{iter_+1:03d}.png"
                        )

                    break

                # If we are aligned, step the whole thing closer by some amount
                # This is based on the pitch - basically
                aligned_once = True
                arm_component = np.cos(wrist_pitch) * self.lift_arm_ratio
                lift_component = np.sin(wrist_pitch) * self.lift_arm_ratio

                arm += arm_component
                lift += lift_component

            # Add these to do some really hacky proportionate control
            px = max(0.25, np.abs(2 * dx / target_mask.shape[1]))
            py = max(0.25, np.abs(2 * dy / target_mask.shape[0]))

            # Move the base and modify the wrist pitch
            # TODO: remove debug code
            # print(f"dx={dx}, dy={dy}, px={px}, py={py}")
            print("base x =", base_x)
            if dx > self.align_x_threshold:
                # Move in x - this means translate the base
                base_x += -self.base_x_step * px
            elif dx < -1 * self.align_x_threshold:
                base_x += self.base_x_step * px
            print("base x =", base_x)
            if dy > self.align_y_threshold:
                # Move in y - this means translate the base
                wrist_pitch += -self.wrist_pitch_step * py
            elif dy < -1 * self.align_y_threshold:
                wrist_pitch += self.wrist_pitch_step * py

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
                -0.5,  # 0.0,
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
                blocking=True,
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

        if self.show_servo_gui and not self.headless_machine:
            cv2.destroyAllWindows()
        return success

    def get_object_xyz(self) -> np.ndarray:
        """Get the object xyz location. If we have a target object, we will use that. Otherwise, we will use the object xyz location that's been manually set.

        Returns:
            np.ndarray: Object xyz location
        """
        if self._object_xyz is None:
            # xyz = self.robot.get_base_pose()
            # xyz[2] = 0.5
            # object_xyz = self.agent.current_object.get_closest_point(xyz)
            # object_xyz = self.agent.current_object.get_center()
            object_xyz = self.agent.current_object.get_median()
        else:
            object_xyz = self._object_xyz
        return object_xyz

    def run(self) -> None:
        self.intro("Grasping the object.")
        self._success = False
        if self.show_object_to_grasp:
            self.show_instance(self.agent.current_object)

        # Clear the observation history
        self.reset()

        assert self.target_object is not None, "Target object must be set before running."

        # open gripper
        self.robot.open_gripper(blocking=True)

        # Now we should be able to see the object if we orient gripper properly
        # Get the end effector pose
        obs = self.robot.get_observation()
        joint_state = self.robot.get_joint_positions()
        model = self.robot.get_robot_model()

        if joint_state[HelloStretchIdx.GRIPPER] < 0.0:
            self.robot.open_gripper(blocking=True)

        # Get the current base pose of the robot
        xyt = self.robot.get_base_pose()

        # Note that these are in the robot's current coordinate frame;
        # they're not global coordinates, so this is ok to use to compute motions.

        # New ee pose = roughly the end of the arm
        object_xyz = self.get_object_xyz()
        relative_object_xyz = point_global_to_base(object_xyz, xyt)

        # Compute the angles necessary
        if self.use_pitch_from_vertical:
            # head_pos = obs.camera_pose[:3, 3]
            obs = self.robot.get_observation()
            joint_state = obs.joint
            model = self.robot.get_robot_model()

            ee_pos, ee_rot = model.manip_fk(joint_state)

            # Convert quaternion to pose
            pose = np.eye(4)
            pose[:3, :3] = R.from_quat(ee_rot).as_matrix()
            pose[:3, 3] = ee_pos

            # Move back 0.3m from grasp coordinate
            delta = np.eye(4)
            delta[2, 3] = -0.3
            pose = np.dot(pose, delta)

            # New ee pose = roughly the end of the arm
            ee_pos = pose[:3, 3]

            # dy = np.abs(head_pos[1] - relative_object_xyz[1])
            # dz = np.abs(head_pos[2] - relative_object_xyz[2])
            dy = np.abs(ee_pos[1] - relative_object_xyz[1])
            dz = np.abs(ee_pos[2] - relative_object_xyz[2])
            pitch_from_vertical = np.arctan2(dy, dz)
        else:
            pitch_from_vertical = 0.0

        # Compute final pregrasp joint state goal and send the robot there
        joint_state[HelloStretchIdx.WRIST_PITCH] = self.offset_from_vertical + pitch_from_vertical
        self.robot.arm_to(joint_state, head=constants.look_at_ee, blocking=True)

        if self.servo_to_grasp:
            # If we try to servo, then do this
            self._success = self.visual_servo_to_object(self.agent.current_object)

        # clear observations
        if self.reset_observation:
            self.agent.reset_object_plans()
            self.agent.get_voxel_map().instances.pop_global_instance(
                env_id=0, global_instance_id=self.agent.current_object.global_id
            )

        # Delete the object
        if self.delete_object_after_grasp:
            voxel_map = self.agent.get_voxel_map()
            if voxel_map is not None:
                voxel_map.delete_instance(self.agent.current_object, assume_explored=False)

        # Say we grasped the object
        if self.talk and self._success:
            self.agent.robot_say(f"I think I grasped the {self.sayable_target_object()}.")

        # Go back to manipulation posture
        self.robot.move_to_manip_posture()

    def pregrasp_open_loop(self, object_xyz: np.ndarray, distance_from_object: float = 0.35):
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

        # End effector should be at most 45 degrees inclined
        rotation = R.from_quat(ee_rot)
        rotation = rotation.as_euler("xyz")

        # Track if the angle to the target object is too large (i.e. it's on the floor)
        print("Rotation", rotation)
        if rotation[1] > np.pi / 4:
            rotation[1] = np.pi / 4
        old_ee_rot = ee_rot
        ee_rot = R.from_euler("xyz", rotation).as_quat()

        vector_to_object = relative_object_xyz - ee_pos
        vector_to_object = vector_to_object / np.linalg.norm(vector_to_object)

        # It should not be more than 45 degrees inclined
        vector_to_object[2] = max(vector_to_object[2], vector_to_object[1])

        print("Absolute object xyz was:", object_xyz)
        print("Relative object xyz was:", relative_object_xyz)
        shifted_object_xyz = relative_object_xyz - (distance_from_object * vector_to_object)
        print("Pregrasp xyz:", shifted_object_xyz)

        # IK
        target_joint_positions, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            shifted_object_xyz, ee_rot, q0=joint_state
        )

        print("Pregrasp joint positions: ")
        print(" - arm: ", target_joint_positions[HelloStretchIdx.ARM])
        print(" - lift: ", target_joint_positions[HelloStretchIdx.LIFT])
        print(" - roll: ", target_joint_positions[HelloStretchIdx.WRIST_ROLL])
        print(" - pitch: ", target_joint_positions[HelloStretchIdx.WRIST_PITCH])
        print(" - yaw: ", target_joint_positions[HelloStretchIdx.WRIST_YAW])

        # get point 10cm from object
        if not success:
            print("Failed to find a valid IK solution.")
            self._success = False
            return
        elif (
            target_joint_positions[HelloStretchIdx.ARM] < -0.05
            or target_joint_positions[HelloStretchIdx.LIFT] < -0.05
        ):
            print(
                f"{self.name}: Target joint state is invalid: {target_joint_positions}. Positions for arm and lift must be positive."
            )
            self._success = False
            return

        # Make sure arm and lift are positive
        target_joint_positions[HelloStretchIdx.ARM] = max(
            target_joint_positions[HelloStretchIdx.ARM], 0
        )
        target_joint_positions[HelloStretchIdx.LIFT] = max(
            target_joint_positions[HelloStretchIdx.LIFT], 0
        )

        # Zero out roll and yaw
        target_joint_positions[HelloStretchIdx.WRIST_YAW] = 0
        target_joint_positions[HelloStretchIdx.WRIST_ROLL] = 0

        # Lift the arm up a bit
        target_joint_positions_lifted = target_joint_positions.copy()
        target_joint_positions_lifted[HelloStretchIdx.LIFT] += self.lift_distance

        print(f"{self.name}: Moving to pre-grasp position.")
        self.robot.arm_to(target_joint_positions, head=constants.look_at_ee, blocking=True)
        print("... done.")

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
        self.robot.arm_to(target_joint_positions_lifted, head=constants.look_at_ee, blocking=True)
        print(f"{self.name}: Return arm to initial configuration.")
        self.robot.arm_to(joint_state, head=constants.look_at_ee, blocking=True)
        print(f"{self.name}: Done.")
        self._success = True
        return

    def was_successful(self) -> bool:
        """Return true if successful"""
        return self._success
