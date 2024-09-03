# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import click
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Mapping and perception
from stretch.core.parameters import Parameters, get_parameters
from stretch.dynav import RobotAgentMDP

# Chat and UI tools
from stretch.utils.point_cloud import numpy_to_pcd, show_point_cloud
from stretch.agent import RobotClient

import cv2

def compute_tilt(camera_xyz, target_xyz):
    '''
        a util function for computing robot head tilts so the robot can look at the target object after navigation
        - camera_xyz: estimated (x, y, z) coordinates of camera
        - target_xyz: estimated (x, y, z) coordinates of the target object
    '''
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))

@click.command()
@click.option("--ip", default='100.108.67.79', type=str)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=-1)
@click.option("--re", default=1, type=int)
@click.option(
    "--input-path",
    type=click.Path(),
    default=None,
    help="Input path with default value 'output.npy'",
)
def main(
    ip,
    manual_wait,
    navigate_home: bool = False,
    explore_iter: int = 5,
    re: int = 1,
    input_path: str = None,
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """
    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    robot = RobotClient(robot_ip = "100.79.44.11")

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    # print(parameters)
    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = None, None
    robot.move_to_nav_posture()
    robot.set_velocity(v = 15., w = 8.)

    print("- Start robot agent with data collection")
    demo = RobotAgentMDP(
        robot, parameters, ip = ip, re = re
    )

    if input_path is None:
        demo.rotate_in_place()
    else:
        demo.image_processor.read_from_pickle(input_path)

    # def keep_looking_around():
    #     while True:
    #         demo.look_around()

    # img_thread = threading.Thread(target=keep_looking_around)
    # img_thread.daemon = True
    # img_thread.start()

    while True:
        mode = input('select mode? E/N/S')
        if mode == 'S':
            break
        if mode == 'E':
            robot.switch_to_navigation_mode()
            for epoch in range(explore_iter):
                print('\n', 'Exploration epoch ', epoch, '\n')
                if not demo.run_exploration():
                    print('Exploration failed! Quitting!')
                    break
        else:
            robot.move_to_nav_posture()
            robot.switch_to_navigation_mode()
            text = input('Enter object name: ')
            point = demo.navigate(text)
            if point is None:
                print('Navigation Failure!')
                continue
            robot.switch_to_navigation_mode()
            xyt = robot.get_base_pose()
            xyt[2] = xyt[2] + np.pi / 2
            robot.navigate_to(xyt, blocking = True)
            cv2.imwrite(text + '.jpg', robot.get_observation().rgb[:, :, [2, 1, 0]])

            if input('You want to run manipulation: y/n') == 'n':
                continue
            camera_xyz = robot.get_head_pose()[:3, 3]
            theta = compute_tilt(camera_xyz, point)
            demo.manipulate(text, theta)
            
            robot.switch_to_navigation_mode()
            if input('You want to run placing: y/n') == 'n':
                continue
            text = input('Enter receptacle name: ')
            point = demo.navigate(text)
            if point is None:
                print('Navigation Failure')
                continue
            robot.switch_to_navigation_mode()
            xyt = robot.get_base_pose()
            xyt[2] = xyt[2] + np.pi / 2
            robot.navigate_to(xyt, blocking = True)
            cv2.imwrite(text + '.jpg', robot.get_observation().rgb[:, :, [2, 1, 0]])
        
            if input('You want to run placing: y/n') == 'n':
                continue
            camera_xyz = robot.get_head_pose()[:3, 3]
            theta = compute_tilt(camera_xyz, point)
            demo.place(text, theta)


if __name__ == "__main__":
    main()
