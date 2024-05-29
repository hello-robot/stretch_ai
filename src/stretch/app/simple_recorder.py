import click
import numpy as np

from stretch.agent import HomeRobotZmqClient
from stretch.core import Parameters, get_parameters


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

    last_step = None
    observations = []
    try:
        # Loop and read in messages
        # Write out RGB+D to file
        while True:
            obs = robot.get_observation()
            # Add all unique observations
            if obs is not None:
                print(f"Step: {obs['step']}")
                if obs["step"] != last_step:
                    last_step = obs["step"]
                    observations.append(obs)
    finally:
        # Disconnect from the robot
        robot.stop()
        breakpoint()


if __name__ == "__main__":
    main()
