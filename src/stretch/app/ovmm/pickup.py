#!/usr/bin/env python3

import click

from stretch.agent.zmq_client import HomeRobotZmqClient


@click.command()
@click.option("--robot_ip", default="192.168.1.15", help="IP address of the robot")
@click.option("--recv_port", default=4401, help="Port to receive messages from the robot")
@click.option("--send_port", default=4402, help="Port to send messages to the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
def main(
    robot_ip: str = "192.168.1.15",
    recv_port: int = 4401,
    send_port: int = 4402,
    local: bool = False,
):
    """Set up the robot, create a task plan, and execute it."""
    # Create robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        recv_port=recv_port,
        send_port=send_port,
        use_remote_computer=(not local),
    )


if __name__ == "__main__":
    main()
