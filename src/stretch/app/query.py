# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

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
import stretch.utils.logger as logger
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import Parameters, RobotClient, get_parameters
from stretch.perception import create_semantic_sensor
from stretch.utils.dummy_stretch_client import DummyStretchClient


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--recv_port", default=4401, help="Port to receive observations on")
@click.option("--send_port", default=4402, help="Port to send actions to on the robot")
@click.option("--robot_ip", default="")
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
@click.option(
    "--all-matches",
    is_flag=True,
    help="Find all objects with a similarity to the query above some threshold",
)
# This threshold seems to work ok for Siglip - will not work for e.g. CLIP
@click.option("--threshold", default=0.05, help="Threshold for similarity when using --all-matches")
@click.option(
    "--stationary",
    is_flag=True,
    help="Don't move the robot to the instance, if using real robot instead of offline data",
)
def main(
    device_id: int = 0,
    verbose: bool = True,
    parameter_file: str = "config/default_planner.yaml",
    local: bool = True,
    recv_port: int = 4401,
    send_port: int = 4402,
    robot_ip: str = "",
    reset: bool = False,
    explore_iter: int = 0,
    output_filename: str = "stretch_output",
    spin: bool = False,
    write_instance_images: bool = False,
    input_file: str = "",
    frame: int = -1,
    text: str = "",
    yes: bool = False,
    stationary: bool = False,
    all_matches: bool = False,
    threshold: float = 0.5,
):

    print("- Load parameters")
    parameters = get_parameters(parameter_file)
    _, semantic_sensor = create_semantic_sensor(
        device_id=device_id,
        verbose=verbose,
        category_map_file=parameters["open_vocab_category_map_file"],
    )

    if len(input_file) == 0 or input_file is None:
        real_robot = True
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
    else:
        real_robot = False
        dummy_robot = DummyStretchClient()
        agent = RobotAgent(dummy_robot, parameters, semantic_sensor)
        agent.voxel_map.read_from_pickle(input_file, num_frames=frame)

    try:
        if len(text) == 0:
            # Read text from command line
            text = input("Enter text to encode, empty to quit: ")
            while len(text) > 0:
                # Get the best instance using agent's API
                print(f"Finding best image(s) for '{text}'")
                if all_matches:
                    _, instances = agent.get_instances_from_text(text, threshold=threshold)
                else:
                    _, instance = agent.get_instance_from_text(text)
                    instances = [instance]

                if len(instances) == 0:
                    logger.error("No matches found for query:", text)
                else:
                    for instance in instances:
                        instance.show_best_view(title=text)

                        if real_robot and not stationary:
                            # Confirm before moving
                            if not yes:
                                confirm = input("Move to instance? [y/n]: ")
                                if confirm != "y":
                                    print("Exiting...")
                                    sys.exit(0)
                            print("Moving to instance...")
                            break

                # Get a new query
                text = input("Enter text to encode, empty to quit: ")
        else:
            # Get the best instance using agent's API
            if all_matches:
                instances = agent.get_instances_from_text(text, threshold=threshold)
            else:
                _, instance = agent.get_instance_from_text(text)

            if len(instances) == 0:
                logger.error("No matches found for query")
                return

            # Show the best view of the detected instance
            instance.show_best_view()

            if real_robot and not stationary:
                # Confirm before moving
                if not yes:
                    confirm = input("Move to instance? [y/n]: ")
                    if confirm != "y":
                        print("Exiting...")
                        sys.exit(0)
                print("Moving to instance...")
    except KeyboardInterrupt:
        # Stop the robot now
        robot.stop()
        sys.exit(0)

    # Move to the instance if we are on the real robot and not told to stay stationary
    if not stationary:
        # Move to the instance
        # Note: this is a blocking call
        # Generates a motion plan based on what we can see
        agent.move_to_instance(instance)

    # At the end...
    robot.stop()

    # Debugging: write out images of instances that you saw
    if write_instance_images:
        agent.save_instance_images(".", verbose=True)


if __name__ == "__main__":
    main()
