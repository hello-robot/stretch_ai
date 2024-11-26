# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import click

from stretch.agent.task.dynamem import DynamemTaskExecutor
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core.parameters import get_parameters


@click.command()
# by default you are running these codes on your workstation, not on your robot.
@click.option("--server_ip", "--server-ip", default="127.0.0.1", type=str)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=3)
@click.option("--re", default=3, type=int, help="Choose between stretch RE1, RE2, RE3")
@click.option("--method", default="dynamem", type=str)
@click.option("--mode", default="", type=click.Choice(["navigation", "manipulation", "save", ""]))
@click.option("--env", default=1, type=int)
@click.option("--test", default=1, type=int)
@click.option(
    "--visual_servo",
    "--vs",
    "-V",
    "--visual-servo",
    default=False,
    is_flag=True,
    help="Use visual servoing grasp",
)
@click.option(
    "--robot_ip", type=str, default="", help="Robot IP address (leave empty for saved default)"
)
@click.option("--target_object", type=str, default=None, help="Target object to grasp")
@click.option("--target_receptacle", type=str, default=None, help="Target receptacle to place")
@click.option("--skip_confirmations", "--yes", "-y", is_flag=True, help="Skip many confirmations")
@click.option(
    "--input-path",
    type=click.Path(),
    default=None,
    help="Input path with default value 'output.npy'",
)
@click.option(
    "--match-method", "--match_method", type=click.Choice(["class", "feature"]), default="feature"
)
@click.option("--device_id", default=0, type=int, help="Device ID for semantic sensor")
def main(
    server_ip,
    manual_wait,
    navigate_home: bool = False,
    explore_iter: int = 3,
    re: int = 1,
    mode: str = "navigation",
    method: str = "dynamem",
    env: int = 1,
    test: int = 1,
    input_path: str = None,
    robot_ip: str = "",
    visual_servo: bool = False,
    skip_confirmations: bool = False,
    device_id: int = 0,
    target_object: str = None,
    target_receptacle: str = None,
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")

    print("- Create robot client")
    robot = HomeRobotZmqClient(robot_ip=robot_ip)

    print("- Create task executor")
    task_executor = DynamemTaskExecutor(
        robot, parameters, visual_servo=visual_servo, match_method=kwargs["match_method"],
        device_id=device_id,
    )
    task_executor.run(target_object=target_object, target_receptacle=target_receptacle)


if __name__ == "__main__":
    main()
