import time
import timeit
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from stretch.agent.base import ManagedOperation
from stretch.core.interfaces import Observations
from stretch.mapping.instance import Instance
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.geometry import point_global_to_base
from stretch.utils.gripper import GripperArucoDetector


class GraspObjectOperation(ManagedOperation):
    """Move the robot to grasp, using the end effector camera."""

    use_pitch_from_vertical: bool = True
    lift_distance: float = 0.2
    servo_to_grasp: bool = False
    _success: bool = False

    # Debugging UI elements
    show_object_to_grasp: bool = False
    show_servo_gui: bool = True

    # Thresholds for centering on object
    align_x_threshold: int = 15
    align_y_threshold: int = 10

    # Visual servoing config
    track_image_center: bool = False
    gripper_aruco_detector: GripperArucoDetector = None
    min_points_to_approach: int = 100
    detected_center_offset_y: int = -40
    lift_arm_ratio: float = 0.1
    base_x_step: float = 0.12
    wrist_pitch_step: float = 0.1
    median_distance_when_grasping: float = 0.175
    percentage_of_image_when_grasping: float = 0.2

    # Timing issues
    expected_network_delay = 0.2
    open_loop: bool = False

    def can_start(self):
        """Grasping can start if we have a target object picked out, and are moving to its instance, and if the robot is ready to begin manipulation."""
        return self.manager.current_object is not None and self.robot.in_manipulation_mode()

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
        class_mask = servo.semantic == instance.get_category_id()
        instance_mask = servo.instance
        target_mask = None
        target_mask_pts = float("-inf")
        maximum_overlap_mask = None
        maximum_overlap_pts = float("-inf")
        center_x, center_y = center
        for iid in np.unique(instance_mask):
            current_instance_mask = instance_mask == iid

            # If we are centered on the mask and it's the right class, just go for it
            if class_mask[center_y, center_x] > 0 and current_instance_mask[center_y, center_x] > 0:
                # This is the correct one - it's centered and the right class. Just go there.
                print("!!! CENTERED ON THE RIGHT OBJECT !!!")
                return current_instance_mask

            # Option 2 - try to find the map that most overlapped with what we were just trying to grasp
            # This is in case we are losing track of particular objects and getting classes mixed up
            if prev_mask is not None:
                # Find the mask with the most points
                mask = np.bitwise_and(current_instance_mask, prev_mask)
                num_pts = sum(mask.flatten())

                if num_pts > maximum_overlap_pts:
                    maximum_overlap_pts = num_pts
                    maximum_overlap_mask = mask

            # Simply find the mask with the most points
            mask = np.bitwise_and(instance_mask == iid, class_mask)
            num_pts = sum(mask.flatten())
            if num_pts > target_mask_pts:
                target_mask = mask
                target_mask_pts = num_pts

        if maximum_overlap_pts > self.min_points_to_approach:
            return maximum_overlap_mask
        elif target_mask is not None:
            return target_mask
        else:
            return prev_mask

    def _grasp(self) -> bool:
        """Helper function to close gripper around object."""
        self.cheer("Grasping object!")
        self.robot.close_gripper(blocking=True)
        time.sleep(2.0)

        # Get a joint state for the object
        joint_state = self.robot.get_joint_state()

        # Lifted joint state
        lifted_joint_state = joint_state.copy()
        lifted_joint_state[HelloStretchIdx.LIFT] += 0.2
        self.robot.arm_to(lifted_joint_state, blocking=True)
        time.sleep(2.0)
        return True

    def visual_servo_to_object(self, instance: Instance, max_duration: float = 120.0) -> bool:
        """Use visual servoing to grasp the object."""

        self.intro(f"Visual servoing to grasp object {instance.global_id} {instance.category_id=}.")
        if self.show_servo_gui:
            self.warn("If you want to stop the visual servoing with the GUI up, press 'q'.")

        t0 = timeit.default_timer()
        aligned_once = False
        prev_target_mask = None
        success = False

        # Track the fingertips using aruco markers
        if self.gripper_aruco_detector is None:
            self.gripper_aruco_detector = GripperArucoDetector()

        # Main loop - run unless we time out, blocking.
        while timeit.default_timer() - t0 < max_duration:

            # Get servo observation
            servo = self.robot.get_servo_observation()
            joint_state = self.robot.get_joint_state()

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

            # Run semantic segmentation on it
            servo = self.agent.semantic_sensor.predict(servo, ee=True)
            target_mask = self.get_target_mask(
                servo, instance, prev_mask=prev_target_mask, center=(center_x, center_y)
            )
            target_mask_pts = sum(target_mask.flatten())

            # Get depth
            center_depth = servo.ee_depth[center_y, center_x] / 1000

            # Compute the center of the mask in image coords
            mask = target_mask
            mask_pts = np.argwhere(mask)
            mask_center = mask_pts.mean(axis=0)

            if np.isnan(mask_center).any():
                self.error("Mask center is NaN. This is a problem. Points in mask:", mask_pts)
                continue

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

            # If we have a target mask, compute the median depth of the object
            # Otherwise we will just try to grasp if we are close enough - assume we lost track!
            if target_mask is not None and target_mask_pts > self.min_points_to_approach:
                object_depth = servo.ee_depth[target_mask]
                median_object_depth = np.median(servo.ee_depth[target_mask]) / 1000
            else:
                print("detected classes:", np.unique(servo.semantic))
                if center_depth < self.median_distance_when_grasping:
                    success = self._grasp()
                continue

            dx, dy = mask_center[1] - center_x, mask_center[0] - center_y

            # Is the center of the image part of the target mask or not?
            center_in_mask = target_mask[int(center_y), int(center_x)] > 0

            # Since we were able to detect it, copy over the target mask
            prev_target_mask = target_mask

            print()
            print("----- STEP VISUAL SERVOING -----")
            print("cur x =", base_x)
            print(" lift =", lift)
            print("  arm =", arm)
            print("pitch =", wrist_pitch)
            print(f"base_x={base_x}, wrist_pitch={wrist_pitch}, dx={dx}, dy={dy}")
            print(f"Median distance to object is {median_object_depth}.")
            print(f"Center distance to object is {center_depth}.")
            print("Center in mask?", center_in_mask)
            if center_in_mask and (
                center_depth < self.median_distance_when_grasping
                or median_object_depth < self.median_distance_when_grasping
            ):
                "If there's any chance the object is close enough, we should just try to grasp it." ""
                success = self._grasp()
                break
            elif np.abs(dx) < self.align_x_threshold and np.abs(dy) < self.align_y_threshold:
                # First, check to see if we are close enough to grasp
                if center_depth < self.median_distance_when_grasping:
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

            print("tgt x =", base_x)
            print(" lift =", lift)
            print("  arm =", arm)
            print("pitch =", wrist_pitch)

            # breakpoint()
            self.robot.arm_to([base_x, lift, arm, 0, wrist_pitch, 0], blocking=True)
            time.sleep(self.expected_network_delay)

        return success

    def run(self):
        self.intro("Grasping the object.")
        self._success = False
        if self.show_object_to_grasp:
            self.show_instance(self.manager.current_object)

        # Now we should be able to see the object if we orient gripper properly
        # Get the end effector pose
        obs = self.robot.get_observation()
        joint_state = obs.joint
        model = self.robot.get_robot_model()

        if joint_state[HelloStretchIdx.GRIPPER] < 0.0:
            self.robot.open_gripper(blocking=True)

        # Get the current base pose of the robot
        xyt = self.robot.get_base_pose()

        # Note that these are in the robot's current coordinate frame; they're not global coordinates, so this is ok to use to compute motions.
        ee_pos, ee_rot = model.manip_fk(joint_state)
        object_xyz = self.manager.current_object.point_cloud.mean(axis=0)
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
        # arm_cmd = self.robot_model.config_to_manip_command(joint_state)
        self.robot.arm_to(joint_state, blocking=True)

        if self.servo_to_grasp:
            self._success = self.visual_servo_to_object(self.manager.current_object)
        else:
            target_joint_state, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
                relative_object_xyz, ee_rot, q0=joint_state
            )
            target_joint_state[HelloStretchIdx.BASE_X] -= 0.04
            if not success:
                print("Failed to find a valid IK solution.")
                self._success = False
                return
            elif (
                target_joint_state[HelloStretchIdx.ARM] < 0
                or target_joint_state[HelloStretchIdx.LIFT] < 0
            ):
                print(
                    f"{self.name}: Target joint state is invalid: {target_joint_state}. Positions for arm and lift must be positive."
                )
                self._success = False
                return

            # Lift the arm up a bit
            target_joint_state_lifted = target_joint_state.copy()
            target_joint_state_lifted[HelloStretchIdx.LIFT] += self.lift_distance

            # Move to the target joint state
            print(f"{self.name}: Moving to grasp position.")
            self.robot.arm_to(target_joint_state, blocking=True)
            time.sleep(3.0)
            print(f"{self.name}: Closing the gripper.")
            self.robot.close_gripper(blocking=True)
            time.sleep(2.0)
            print(f"{self.name}: Lifting the arm up so as not to hit the base.")
            self.robot.arm_to(target_joint_state_lifted, blocking=False)
            time.sleep(2.0)
            print(f"{self.name}: Return arm to initial configuration.")
            self.robot.arm_to(joint_state, blocking=True)
            time.sleep(3.0)
            print(f"{self.name}: Done.")
            self._success = True

    def was_successful(self):
        """Return true if successful"""
        return self._success
