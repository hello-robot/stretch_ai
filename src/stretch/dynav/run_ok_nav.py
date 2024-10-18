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

from stretch.agent import RobotClient

# Mapping and perception
from stretch.core.parameters import get_parameters
from stretch.dynav import RobotAgentMDP


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


@click.command()
# by default you are running these codes on your workstation, not on your robot.
@click.option("--server_ip", default="", type=str, help="IP address for the MDP agent")
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=-1)
@click.option("--re", default=3, type=int, help="Choose between stretch RE1, RE2, RE3")
@click.option("--method", default="dynamem", type=str)
@click.option("--env", default=1, type=int)
@click.option("--test", default=1, type=int)
@click.option(
    "--robot_ip", type=str, default="", help="Robot IP address (leave empty for saved default)"
)
@click.option(
    "--input-path",
    type=click.Path(),
    default=None,
    help="Input path with default value 'output.npy'",
)
def main(
    server_ip,
    manual_wait,
    navigate_home: bool = False,
    explore_iter: int = 5,
    re: int = 1,
    method: str = "dynamem",
    env: int = 1,
    test: int = 1,
    input_path: str = None,
    robot_ip: str = "",
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """
    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    robot = RobotClient(robot_ip=robot_ip)
    robot.move_to_nav_posture()

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    # print(parameters)
    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = None, None
    robot.move_to_nav_posture()
    robot.set_velocity(v=30.0, w=15.0)

    print("- Start robot agent with data collection")
    demo = RobotAgentMDP(
        robot, parameters, server_ip=server_ip, re=re, env_num=env, test_num=test, method=method
    )

    if input_path is None:
        demo.rotate_in_place()
    else:
        demo.image_processor.read_from_pickle(input_path)

    demo.save()

    # def keep_looking_around():
    #     while True:
    #         # We don't want too many images in our memory
    #         time.sleep(0.8)
    #         if robot.get_six_joints()[2] > 0.7 or not robot.in_navigation_mode():
    #             continue
    #         demo.update()

    # img_thread = threading.Thread(target=keep_looking_around)
    # img_thread.daemon = True
    # img_thread.start()

    while True:
        print("Select mode: E for exploration, N for open-vocabulary navigation, S for save.")
        mode = input("select mode? E/N/S: ")
        mode = mode.upper()
        if mode == "S":
            demo.image_processor.write_to_pickle()
            break
        if mode == "E":
            robot.switch_to_navigation_mode()
            for epoch in range(explore_iter):
                print("\n", "Exploration epoch ", epoch, "\n")
                if not demo.run_exploration():
                    print("Exploration failed! Quitting!")
                    continue
        else:
            text = None
            point = None
            if input("You want to run manipulation: y/n") != "n":
                robot.move_to_nav_posture()
                robot.switch_to_navigation_mode()
                text = input("Enter object name: ")
                point = demo.navigate(text)
                if point is None:
                    print("Navigation Failure!")
                cv2.imwrite(text + ".jpg", robot.get_observation().rgb[:, :, [2, 1, 0]])
                robot.switch_to_navigation_mode()
                xyt = robot.get_base_pose()
                xyt[2] = xyt[2] + np.pi / 2
                robot.navigate_to(xyt, blocking=True)

            if input("You want to run manipulation: y/n") != "n":
                robot.switch_to_manipulation_mode()
                if text is None:
                    text = input("Enter object name: ")
                camera_xyz = robot.get_head_pose()[:3, 3]
                if point is not None:
                    theta = compute_tilt(camera_xyz, point)
                else:
                    theta = -0.6
                demo.manipulate(text, theta)
                robot.look_front()

            text = None
            point = None
            if input("You want to run placing: y/n") != "n":
                robot.switch_to_navigation_mode()
                text = input("Enter receptacle name: ")
                point = demo.navigate(text)
                if point is None:
                    print("Navigation Failure")
                cv2.imwrite(text + ".jpg", robot.get_observation().rgb[:, :, [2, 1, 0]])
                robot.switch_to_navigation_mode()
                xyt = robot.get_base_pose()
                xyt[2] = xyt[2] + np.pi / 2
                robot.navigate_to(xyt, blocking=True)

            if input("You want to run placing: y/n") != "n":
                robot.switch_to_manipulation_mode()
                if text is None:
                    text = input("Enter receptacle name: ")
                camera_xyz = robot.get_head_pose()[:3, 3]
                if point is not None:
                    theta = compute_tilt(camera_xyz, point)
                else:
                    theta = -0.6
                demo.place(text, theta)
                robot.move_to_nav_posture()

            demo.save()


if __name__ == "__main__":
    main()
