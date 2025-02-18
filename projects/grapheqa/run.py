# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import click
from robot_agent import RobotAgent

from stretch.agent.zmq_client import HomeRobotZmqClient

# Mapping and perception
from stretch.core.parameters import get_parameters
from stretch.perception import create_semantic_sensor


@click.command()
# by default you are running these codes on your workstation, not on your robot.
@click.option("--robot_ip", "--robot-ip", default="127.0.0.1", type=str)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=3)
@click.option("--re", default=3, type=int, help="Choose between stretch RE1, RE2, RE3")
@click.option("--method", default="dynamem", type=str)
@click.option("--mode", default="", type=click.Choice(["navigation", "manipulation", "save", ""]))
@click.option("--env", default=1, type=int)
@click.option("--test", default=1, type=int)
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
@click.option("--device_id", default=0, type=int, help="Device ID for semantic sensor")
def main(
    robot_ip,
    manual_wait,
    navigate_home: bool = False,
    explore_iter: int = 3,
    re: int = 1,
    mode: str = "navigation",
    method: str = "dynamem",
    env: int = 1,
    test: int = 1,
    input_path: str = None,
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
    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    robot = HomeRobotZmqClient(robot_ip=robot_ip)

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    # print(parameters)
    # if explore_iter >= 0:
    #     parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = None, None
    robot.move_to_nav_posture()
    robot.set_velocity(v=30.0, w=15.0)

    # Create semantic sensor if visual servoing is enabled
    print("- Create semantic sensor if visual servoing is enabled")
    semantic_sensor = create_semantic_sensor(
        parameters=parameters,
        device_id=device_id,
        verbose=False,
    )
    parameters["encoder"] = None

    print("- Start robot agent with data collection")
    agent = RobotAgent(robot, parameters, semantic_sensor)
    agent.start()

    if input_path is None:
        agent.rotate_in_place()
    else:
        agent.voxel_map.read_from_pickle(input_path)

    # agent.voxel_map.write_to_pickle(None)

    while agent.is_running():

        # If target object and receptacle are provided, set mode to manipulation
        # mode = input("select mode? E/S: ").lower()

        # robot.say("Saving data. Goodbye!")
        # # agent.image_processor.write_to_pickle(None)
        # break

        robot.move_to_nav_posture()
        robot.switch_to_navigation_mode()
        robot.say("Running EQA.")
        for epoch in range(explore_iter):
            print("\n", "Exploration epoch ", epoch, "\n")
            if not agent.run_exploration():
                print("Exploration failed! Quitting!")
                continue


if __name__ == "__main__":
    main()
