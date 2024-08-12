# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import pickle
import time

import click
import matplotlib
import numpy as np

from stretch.agent import HomeRobotZmqClient
from stretch.core import get_parameters

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


@click.command()
@click.option("--robot_ip", default="192.168.1.15", help="IP address of the robot")
@click.option("--recv_port", default=4401, help="Port to receive messages from the robot")
@click.option("--send_port", default=4402, help="Port to send messages to the robot")
@click.option("--local", is_flag=True, help="Is this code running locally on the robot")
@click.option(
    "--parameter_file", default="config/default_planner.yaml", help="Path to parameter file"
)
def main(
    robot_ip: str = "192.168.1.15",
    recv_port: int = 4401,
    send_port: int = 4402,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    verbose: bool = False,
    show_intermediate_maps: bool = False,
    reset: bool = False,
    local: bool = False,
):
    """Set up the robot, create a task plan, and execute it."""
    # Create robot
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        recv_port=recv_port,
        send_port=send_port,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    started = robot.start()
    if not started:
        raise RuntimeError("Cannot make connection to robot")
    time.sleep(1.0)

    # A test
    # robot.move_to_nav_posture()
    # robot.blocking_spin()
    # breakpoint()

    last_seq_id = None
    observations = []
    try:
        # Loop and read in messages
        # Write out RGB+D to file
        while True:
            obs = robot.get_observation()
            # Add all unique observations
            if obs is not None and obs.seq_id != last_seq_id:
                print(f"Step: {obs.seq_id}")
                last_seq_id = obs.seq_id
                observations.append(obs)
    finally:
        # Disconnect from the robot
        robot.stop()

        X, Y = [], []
        theta = []
        for obs in observations:
            X.append(obs.gps[0])
            Y.append(obs.gps[1])
            theta.append(obs.compass)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the (x, y) points
        ax.scatter(X, Y, s=20, color="b")

        # Plot the arrows indicating direction
        Q = ax.quiver(X, Y, np.cos(theta), np.sin(theta), angles="xy", scale_units="xy", scale=0.05)
        ax.quiverkey(Q, 0.9, 0.95, 1, "Direction", labelpos="E", coordinates="figure")

        plt.show()

        print("Saving observations to file")
        with open("observations.pkl", "wb") as f:
            pickle.dump(observations, f)


if __name__ == "__main__":
    main()
