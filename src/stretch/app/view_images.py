#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time

import click
import cv2
import numpy as np

import stretch.utils.logger as logger
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.perception import create_semantic_sensor
from stretch.utils.gripper import GripperArucoDetector
from stretch.utils.image import adjust_gamma


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option(
    "--target_object", type=str, default="shoe", help="Type of object to pick up and move"
)
@click.option("--gamma", type=float, default=1.0, help="Gamma correction factor for EE rgb images")
@click.option(
    "--run_semantic_segmentation", is_flag=True, help="Run semantic segmentation on EE rgb images"
)
@click.option("--segment_ee", is_flag=True, help="Run semantic segmentation on EE rgb images")
@click.option("--aruco", is_flag=True, help="Run aruco detection on EE rgb images")
def main(
    robot_ip: str = "192.168.1.15",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    verbose: bool = False,
    show_intermediate_maps: bool = False,
    reset: bool = False,
    target_object: str = "shoe",
    run_semantic_segmentation: bool = False,
    gamma: float = 1.0,
    segment_ee: bool = False,
    aruco: bool = False,
):
    # Create robot
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
        enable_rerun_server=False,
    )
    if segment_ee:
        run_semantic_segmentation = True
    if run_semantic_segmentation:
        semantic_sensor = create_semantic_sensor(
            parameters,
            device_id=device_id,
            verbose=verbose,
        )
    if aruco:
        aruco_detector = GripperArucoDetector()
    else:
        aruco_detector = None

    print("Starting the robot...")
    robot.start()
    robot.move_to_manip_posture()
    robot.open_gripper()
    time.sleep(2)
    # robot.arm_to([0.0, 0.78, 0.05, 0, -3 * np.pi / 8, 0], blocking=True)
    robot.arm_to([0.0, 0.4, 0.05, 0, -np.pi / 4, 0], blocking=True)

    # Initialize variables
    first_time = True
    warning_ee = False
    colors = {}

    # Loop and read in images
    print("Reading images...")
    while True:
        # Get image from robot
        obs = robot.get_observation()
        if obs is None:
            continue
        if obs.rgb is None:
            continue
        # Low res images used for visual servoing and ML
        servo = robot.get_servo_observation()
        if servo is None:
            print("No servo observation. Skipping.")
            continue
        if servo.ee_rgb is None:
            if not warning_ee:
                logger.warning("No end effector image.")
            warning_ee = True
        if servo.ee_depth is None:
            if not warning_ee:
                logger.warning("No end effector depth image.")
            warning_ee = True

        # First time info
        if first_time:
            print("First observation received. Semantic sensor will be slow the first time.")
            print("Full (slow) observation:")
            print(" - RGB image shape:", repr(obs.rgb.shape))
            print("Servo observation:")
            if servo.ee_rgb is not None:
                print(" - ee rgb shape:", repr(servo.ee_rgb.shape))
                print(" - ee depth shape:", repr(servo.ee_depth.shape))
            print(" - head rgb shape:", repr(servo.rgb.shape))
            print(" - head depth shape:", repr(servo.depth.shape))
            print()
            print("Press 'q' with a window selected to quit.")
            first_time = False

        # Run segmentation if you want
        if servo.ee_rgb is not None:
            servo.ee_rgb = adjust_gamma(servo.ee_rgb, gamma)

        if run_semantic_segmentation:
            # Run the prediction on end effector camera!
            use_ee = segment_ee and servo.ee_rgb is not None
            use_full_obs = False
            if use_full_obs:
                use_ee = False
                _obs = obs
            else:
                _obs = servo
            _obs = semantic_sensor.predict(_obs, ee=use_ee)
            semantic_segmentation = np.zeros(
                (_obs.semantic.shape[0], _obs.semantic.shape[1], 3)
            ).astype(np.uint8)
            for cls in np.unique(_obs.semantic):
                if cls > 0:
                    if cls not in colors:
                        colors[cls] = (np.random.rand(3) * 255).astype(np.uint8)
                    semantic_segmentation[_obs.semantic == cls] = colors[cls]

            # Compose the two images
            alpha = 0.5
            img = _obs.ee_rgb if use_ee else _obs.rgb
            semantic_segmentation = cv2.addWeighted(
                img.copy(), alpha, semantic_segmentation, 1 - alpha, 0
            )

        # Visualize depth
        viz_depth = cv2.normalize(servo.depth, None, 0, 255, cv2.NORM_MINMAX)
        viz_depth = viz_depth.astype(np.uint8)
        viz_depth = cv2.applyColorMap(viz_depth, cv2.COLORMAP_JET)

        # Visualize end effector depth
        if servo.ee_depth is not None:
            viz_ee_depth = cv2.normalize(servo.ee_depth, None, 0, 255, cv2.NORM_MINMAX)
            viz_ee_depth = viz_ee_depth.astype(np.uint8)
            viz_ee_depth = cv2.applyColorMap(viz_ee_depth, cv2.COLORMAP_JET)

        # This is the head image
        image = obs.rgb

        # Display image and semantic segmentation
        # Convert rgb to bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("head camera image", image)
        servo_head_rgb = cv2.cvtColor(servo.rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("servo: head camera image", servo_head_rgb)

        if servo.ee_rgb is not None:
            servo_ee_rgb = cv2.cvtColor(servo.ee_rgb, cv2.COLOR_RGB2BGR)
            if aruco_detector is not None:
                (
                    servo_corners,
                    servo_ids,
                    servo_ee_rgb,
                ) = aruco_detector.detect_and_draw_aruco_markers(servo_ee_rgb)
            cv2.imshow("servo: ee camera image", servo_ee_rgb)
            cv2.imshow("servo: ee depth image", viz_ee_depth)

        cv2.imshow("servo: head depth image", viz_depth)
        if run_semantic_segmentation:
            cv2.imshow("semantic_segmentation", semantic_segmentation)

        # Break if 'q' is pressed
        res = cv2.waitKey(1) & 0xFF  # 0xFF is a mask to get the last 8 bits
        if res == ord("q"):
            break

    robot.stop()


if __name__ == "__main__":
    main()
