#!/usr/bin/env python3

import time

import click
import cv2
import numpy as np

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import Parameters, get_parameters
from stretch.perception import create_semantic_sensor, get_encoder
from stretch.utils.image import adjust_gamma


@click.command()
@click.option("--robot_ip", default="192.168.1.15", help="IP address of the robot")
@click.option("--recv_port", default=4401, help="Port to receive messages from the robot")
@click.option("--send_port", default=4402, help="Port to send messages to the robot")
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
@click.option("--gamma", type=float, default=2.0, help="Gamma correction factor for EE rgb images")
@click.option(
    "--run_semantic_segmentation", is_flag=True, help="Run semantic segmentation on EE rgb images"
)
def main(
    robot_ip: str = "192.168.1.15",
    recv_port: int = 4401,
    send_port: int = 4402,
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    verbose: bool = False,
    show_intermediate_maps: bool = False,
    reset: bool = False,
    target_object: str = "shoe",
    run_semantic_segmentation: bool = False,
    gamma: float = 1.5,
):
    # Create robot
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        recv_port=recv_port,
        send_port=send_port,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    if run_semantic_segmentation:
        _, semantic_sensor = create_semantic_sensor(
            device_id=device_id,
            verbose=verbose,
            category_map_file=parameters["open_vocab_category_map_file"],
        )

    print("Starting the robot...")
    robot.start()
    robot.move_to_manip_posture()
    robot.open_gripper()
    time.sleep(2)
    robot.arm_to([0.0, 0.78, 0.05, 0, -3 * np.pi / 8, 0], blocking=True)

    # Initialize variables
    first_time = True

    # Loop and read in images
    print("Reading images...")
    while True:
        # Get image from robot
        obs = robot.get_observation()
        if obs is None:
            continue
        # Low res images used for visual servoing and ML
        servo = robot.get_servo_observation()

        # First time info
        if first_time:
            print("First observation received. Semantic sensor will be slow the first time.")
            print("Full (slow) observation:")
            print(" - RGB image shape:", repr(obs.rgb.shape))
            print("Servo observation:")
            print(" - ee rgb shape:", repr(servo["ee_cam"]["color_image"].shape))
            print(" - ee depth shape:", repr(servo["ee_cam"]["depth_image"].shape))
            print(" - head rgb shape:", repr(servo["head_cam"]["color_image"].shape))
            print(" - head depth shape:", repr(servo["head_cam"]["depth_image"].shape))
            print()
            print("Press 'q' to quit.")
            first_time = False

        # Run segmentation if you want
        if run_semantic_segmentation:
            obs = semantic_sensor.predict(obs)

        # This is the head image
        image = obs.rgb
        semantic_segmentation = obs.semantic
        # Get semantic segmentation from image
        # Display image and semantic segmentation
        # Convert rgb to bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("head camera image", image)
        servo_head_rgb = cv2.cvtColor(servo["head_cam"]["color_image"], cv2.COLOR_RGB2BGR)
        cv2.imshow("servo: ee camera image", servo_head_rgb)
        servo_ee_rgb = cv2.cvtColor(servo["ee_cam"]["color_image"], cv2.COLOR_RGB2BGR)
        servo_ee_rgb = adjust_gamma(servo_ee_rgb, gamma)
        cv2.imshow("servo: head camera image", servo_ee_rgb)
        if run_semantic_segmentation:
            cv2.imshow("semantic_segmentation", semantic_segmentation)

        # Break if 'q' is pressed
        res = cv2.waitKey(1) & 0xFF  # 0xFF is a mask to get the last 8 bits
        if res == ord("q"):
            break

    robot.stop()


if __name__ == "__main__":
    main()
