# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import click
import cv2
import numpy as np

from stretch.agent.operations import GraspObjectOperation
from stretch.agent.robot_agent_dynamem import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient

# Mapping and perception
from stretch.core.parameters import get_parameters
from stretch.perception import create_semantic_sensor


def compute_tilt(camera_xyz, target_xyz):
    """
    a util function for computing robot head tilts so the robot can look at the target object after navigation
    - camera_xyz: estimated (x, y, z) coordinates of camera
    - target_xyz: estimated (x, y, z) coordinates of the target object
    """
    if not isinstance(camera_xyz, np.ndarray):
        camera_xyz = np.array(camera_xyz)
    if not isinstance(target_xyz, np.ndarray):
        target_xyz = np.array(target_xyz)
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))


def get_mode(mode: str) -> str:

    if mode == "navigation":
        return "N"
    elif mode == "explore":
        return "E"
    elif mode == "manipulation":
        return "manipulation"
    elif mode == "save":
        return "S"
    else:
        mode = None
        print("Select mode: E for exploration, N for open-vocabulary navigation, S for save.")
        while mode is None:
            mode = input("select mode? E/N/S: ")
            if mode == "E" or mode == "N" or mode == "S":
                break
            else:
                print("Invalid mode. Please select again.")
        return mode


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
    "--visual_servo", "--vs", "-V", default=False, is_flag=True, help="Use visual servoing grasp"
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
    if visual_servo:
        semantic_sensor = create_semantic_sensor(
            parameters=parameters,
            device_id=device_id,
            verbose=False,
        )
    else:
        parameters["encoder"] = None
        semantic_sensor = None

    print("- Start robot agent with data collection")
    agent = RobotAgent(robot, parameters, semantic_sensor)
    agent.start()

    if visual_servo:
        grasp_object = GraspObjectOperation(
            "grasp_the_object",
            agent,
        )
    else:
        grasp_object = None

    if input_path is None:
        agent.rotate_in_place()
    else:
        agent.image_processor.read_from_pickle(input_path)

    agent.save()

    while agent.is_running():
        mode = get_mode(mode)
        mode = mode.upper()
        if mode == "S":
            agent.image_processor.write_to_pickle()
            break
        if mode == "E":
            robot.switch_to_navigation_mode()
            for epoch in range(explore_iter):
                print("\n", "Exploration epoch ", epoch, "\n")
                if not agent.run_exploration():
                    print("Exploration failed! Quitting!")
                    continue
        else:
            text = None
            point = None
            if skip_confirmations or input("You want to run manipulation? (y/n): ") != "n":
                robot.move_to_nav_posture()
                robot.switch_to_navigation_mode()
                if target_object is not None:
                    text = target_object
                else:
                    text = input("Enter object name: ")
                point = agent.navigate(text)
                if point is None:
                    print("Navigation Failure!")
                cv2.imwrite(text + ".jpg", robot.get_observation().rgb[:, :, [2, 1, 0]])
                robot.switch_to_navigation_mode()
                xyt = robot.get_base_pose()
                xyt[2] = xyt[2] + np.pi / 2
                robot.navigate_to(xyt, blocking=True)

            if skip_confirmations or input("You want to run manipulation? (y/n): ") != "n":
                robot.switch_to_manipulation_mode()
                if text is None:
                    text = input("Enter object name: ")
                camera_xyz = robot.get_head_pose()[:3, 3]
                if point is not None:
                    theta = compute_tilt(camera_xyz, point)
                else:
                    theta = -0.6

                # Grasp the object using operation if it's available
                if grasp_object is not None:
                    print("Using operation to grasp object:", text)
                    print(" - Point:", point)
                    print(" - Theta:", theta)
                    grasp_object(
                        target_object=text,
                        object_xyz=point,
                        match_method="feature",
                        show_object_to_grasp=False,
                        show_servo_gui=True,
                        delete_object_after_grasp=False,
                    )
                else:
                    # Otherwise, use the agent's manipulation method
                    # This is from OK Robot
                    print("Using agent to grasp object:", text)
                    agent.manipulate(text, theta, skip_confirmation=skip_confirmations)
                robot.look_front()

            text = None
            point = None
            if skip_confirmations or input("You want to run placement? (y/n): ") != "n":
                robot.switch_to_navigation_mode()
                if target_receptacle is not None:
                    text = target_receptacle
                else:
                    text = input("Enter receptacle name: ")
                point = agent.navigate(text)
                if point is None:
                    print("Navigation Failure")
                cv2.imwrite(text + ".jpg", robot.get_observation().rgb[:, :, [2, 1, 0]])
                robot.switch_to_navigation_mode()
                xyt = robot.get_base_pose()
                xyt[2] = xyt[2] + np.pi / 2
                robot.navigate_to(xyt, blocking=True)

            if skip_confirmations or input("You want to run placement? (y/n): ") != "n":
                robot.switch_to_manipulation_mode()
                if text is None:
                    text = input("Enter receptacle name: ")
                camera_xyz = robot.get_head_pose()[:3, 3]
                if point is not None:
                    theta = compute_tilt(camera_xyz, point)
                else:
                    theta = -0.6
                agent.place(text, theta)
                robot.move_to_nav_posture()

            agent.save()


if __name__ == "__main__":
    main()
