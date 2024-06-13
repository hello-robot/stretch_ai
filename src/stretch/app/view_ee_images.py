#!/usr/bin/env python3

import click
import cv2
import numpy as np

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import Parameters, get_parameters
from stretch.perception import create_semantic_sensor, get_encoder


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
@click.option(
    "--parameter_file", default="config/default_planner.yaml", help="Path to parameter file"
)
@click.option(
    "--target_object", type=str, default="shoe", help="Type of object to pick up and move"
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
    do_semantic_segmentation: bool = False,
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
    if do_semantic_segmentation:
        _, semantic_sensor = create_semantic_sensor(
            device_id=device_id,
            verbose=verbose,
            category_map_file=parameters["open_vocab_category_map_file"],
        )

    print("Starting the robot...")
    robot.start()
    robot.move_to_manip_posture()
    robot.arm_to([0.0, 0.78, 0.05, 0, -np.pi / 4, 0], blocking=True)

    # Initialize variables
    first_time = True

    # Loop and read in images
    print("Reading images...")
    while True:
        # Get image from robot
        obs = robot.get_observation()
        if obs is None:
            continue
        if first_time:
            print("First observation received. Semantic sensor will be slow the first time.")
            print("RGB image shape:", repr(obs.rgb.shape))
            print()
            print("Press 'q' to quit.")
            first_time = False
        if do_semantic_segmentation:
            obs = semantic_sensor.predict(obs)

        # This is the head image
        image = obs.rgb
        semantic_segmentation = obs.semantic
        # Get semantic segmentation from image
        # Display image and semantic segmentation
        # Convert rgb to bgr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("head camera image", image)
        if do_semantic_segmentation:
            cv2.imshow("semantic_segmentation", semantic_segmentation)

        # Break if 'q' is pressed
        res = cv2.waitKey(1) & 0xFF  # 0xFF is a mask to get the last 8 bits
        if res == ord("q"):
            break

    robot.stop()


if __name__ == "__main__":
    main()
