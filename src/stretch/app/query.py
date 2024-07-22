# (c) 2024 Hello Robot by Chris Paxton
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import pickle
import sys
import time
import timeit
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np

# Mapping and perception
import stretch.utils.depth as du
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import Parameters, RobotClient, get_parameters
from stretch.perception import create_semantic_sensor
from stretch.utils.dummy_stretch_client import DummyStretchClient


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--recv_port", default=4401, help="Port to receive observations on")
@click.option("--send_port", default=4402, help="Port to send actions to on the robot")
@click.option("--robot_ip", default="192.168.1.15")
@click.option("--output-filename", default="stretch_output", type=str)
@click.option("--explore-iter", default=0)
@click.option("--spin", default=False, is_flag=True)
@click.option("--reset", is_flag=True)
@click.option(
    "--input_file", default="", type=str, help="Path to input file used instead of robot data"
)
@click.option(
    "--write-instance-images",
    default=False,
    is_flag=True,
    help="write out images of every object we found",
)
@click.option("--parameter-file", default="default_planner.yaml")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option("--frame", default=-1, help="Final frame to read from input file")
@click.option("--text", default="", help="Text to encode")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation")
def main(
    device_id: int = 0,
    verbose: bool = True,
    parameter_file: str = "config/default_planner.yaml",
    local: bool = True,
    recv_port: int = 4401,
    send_port: int = 4402,
    robot_ip: str = "192.168.1.15",
    reset: bool = False,
    explore_iter: int = 0,
    output_filename: str = "stretch_output",
    spin: bool = False,
    write_instance_images: bool = False,
    input_file: str = "",
    frame: int = -1,
    text: str = "",
    yes: bool = False,
):

    print("- Load parameters")
    parameters = get_parameters(parameter_file)
    _, semantic_sensor = create_semantic_sensor(
        device_id=device_id,
        verbose=verbose,
        category_map_file=parameters["open_vocab_category_map_file"],
    )

    if len(input_file) == 0 or input_file is None:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        output_pkl_filename = output_filename + "_" + formatted_datetime + ".pkl"

        robot = HomeRobotZmqClient(
            robot_ip=robot_ip,
            recv_port=recv_port,
            send_port=send_port,
            use_remote_computer=(not local),
            parameters=parameters,
        )
        robot.move_to_nav_posture()
        agent = RobotAgent(robot, parameters, semantic_sensor)

        if reset:
            agent.move_closed_loop([0, 0, 0], max_time=60.0)

        if spin:
            # Rotate and capture many frames
            agent.rotate_in_place(steps=8, visualize=False)
        else:
            # Just capture a single frame
            agent.update()

        if explore_iter > 0:
            raise NotImplementedError("Exploration not implemented yet")

        # Save out file
        if len(output_pkl_filename) > 0:
            print(f"Write pkl to {output_pkl_filename}...")
            agent.voxel_map.write_to_pickle(output_pkl_filename)

        # At the end...
        robot.stop()
    else:
        dummy_robot = DummyStretchClient()
        agent = RobotAgent(dummy_robot, parameters, semantic_sensor)
        agent.voxel_map.read_from_pickle(input_file, num_frames=frame)

    if len(text) == 0:
        # Read text from command line
        text = input("Enter text to encode, empty to quit: ")
        while len(text) > 0:
            # Get the best instance using agent's API
            print("Best image for:", text)
            _, instance = agent.get_instance_from_text(text)

            # Show the best view of the detected instance
            instance.show_best_view(title=text)
            text = input("Enter text to encode, empty to quit: ")
    else:
        # Get the best instance using agent's API
        _, instance = agent.get_instance_from_text(text)

        # Show the best view of the detected instance
        instance.show_best_view()

    # Go to the instance view
    agent.move_to_instance_view(instance)

    # Debugging: write out images of instances that you saw
    if write_instance_images:
        agent.save_instance_images(".", verbose=True)


if __name__ == "__main__":
    main()
