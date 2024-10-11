# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# (c) 2024 Hello Robot by Chris Paxton
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
from pathlib import Path
from typing import Optional

import click
import numpy as np

import stretch.app.dex_teleop.dex_teleop_utils as dt_utils

# Mapping and perception
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.app.lfd.ros2_lfd_leader import ROS2LfdLeader
from stretch.core import AbstractRobotClient, Parameters, get_parameters
from stretch.perception import create_semantic_sensor


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--robot_ip", default="")
@click.option("--rate", default=5, type=int)
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--output-filename", default="stretch_output", type=str)
@click.option("--show-intermediate-maps", default=False, is_flag=True)
@click.option("--show-final-map", default=False, is_flag=True)
@click.option("--show-paths", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=-1)
@click.option("--navigate-home", default=False, is_flag=True)
@click.option("--force-explore", default=False, is_flag=True)
@click.option("--no-manip", default=False, is_flag=True)
@click.option(
    "--write-instance-images",
    default=False,
    is_flag=True,
    help="write out images of every object we found",
)
@click.option("--parameter-file", default="default_planner.yaml")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
def main(
    rate,
    visualize,
    manual_wait,
    output_filename,
    navigate_home: bool = True,
    device_id: int = 0,
    verbose: bool = True,
    show_intermediate_maps: bool = False,
    show_final_map: bool = False,
    show_paths: bool = False,
    random_goals: bool = True,
    force_explore: bool = False,
    no_manip: bool = False,
    explore_iter: int = 10,
    write_instance_images: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    local: bool = True,
    robot_ip: str = "192.168.1.15",
    reset: bool = False,
    **kwargs,
):

    print("- Load parameters")
    parameters = get_parameters(parameter_file)

    MANIP_MODE_CONTROLLED_JOINTS = dt_utils.get_teleop_controlled_joints("base_x")
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
        manip_mode_controlled_joints=MANIP_MODE_CONTROLLED_JOINTS,
    )
    # Call demo_main with all the arguments
    demo_main(
        robot,
        parameters=parameters,
        rate=rate,
        visualize=visualize,
        manual_wait=manual_wait,
        output_filename=output_filename,
        navigate_home=navigate_home,
        device_id=device_id,
        verbose=verbose,
        show_intermediate_maps=show_intermediate_maps,
        show_final_map=show_final_map,
        show_paths=show_paths,
        random_goals=random_goals,
        force_explore=force_explore,
        no_manip=no_manip,
        explore_iter=explore_iter,
        write_instance_images=write_instance_images,
        parameter_file=parameter_file,
        reset=reset,
        **kwargs,
    )


def demo_main(
    robot: AbstractRobotClient,
    rate,
    visualize,
    manual_wait,
    output_filename,
    navigate_home: bool = True,
    device_id: int = 0,
    verbose: bool = True,
    show_intermediate_maps: bool = False,
    show_final_map: bool = False,
    show_paths: bool = False,
    random_goals: bool = True,
    force_explore: bool = False,
    no_manip: bool = False,
    explore_iter: int = 10,
    write_instance_images: bool = False,
    parameters: Optional[Parameters] = None,
    parameter_file: str = "config/default.yaml",
    reset: bool = False,
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        show_intermediate_maps(bool): show maps as we explore
        show_final_map(bool): show the final 3d map after moving around and mapping the world
        show_paths(bool): display paths after planning
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """
    cabinet_policy_path = f"/lerobot/outputs/train/2024-07-28/17-34-36_stretch_real_diffusion_default/checkpoints/100000/pretrained_model"
    pickup_policy_path = f"/lerobot/outputs/train/2024-08-07/18-55-17_stretch_real_diffusion_default/checkpoints/100000/pretrained_model"

    cabinet_leader = ROS2LfdLeader(
        robot=robot,
        verbose=False,
        data_dir="./data",
        user_name="Jensen",
        task_name="open_cabinet",
        env_name="kitchen",
        save_images=False,
        teleop_mode="base_x",
        record_success=False,
        policy_name="diffusion",
        policy_path=cabinet_policy_path,
        device="cuda",
        force_execute=True,
        disable_recording=True,
    )

    pickup_leader = ROS2LfdLeader(
        robot=robot,
        verbose=False,
        data_dir="./data",
        user_name="Jensen",
        task_name="pickup_all_purpose",
        env_name="kitchen",
        save_images=False,
        teleop_mode="base_x",
        record_success=False,
        policy_name="diffusion",
        policy_path=pickup_policy_path,
        device="cuda",
        force_execute=True,
        disable_recording=True,
    )

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_pkl_filename = output_filename + "_" + formatted_datetime + ".pkl"

    if parameters is None:
        print("- Load parameters")
        parameters = get_parameters(parameter_file)
        print(parameters)

    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    print("- Connect to Stretch")

    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = parameters.get_task_goals()

    if write_instance_images:
        print("- Create semantic sensor based on detic")
        _, semantic_sensor = create_semantic_sensor(
            device_id=device_id,
            verbose=verbose,
            category_map_file=parameters["open_vocab_category_map_file"],
        )
    else:
        semantic_sensor = None

    print("- Start robot agent")

    pos_err_threshold = parameters["trajectory_pos_err_threshold"]
    rot_err_threshold = parameters["trajectory_rot_err_threshold"]

    # input_path = "kitchen_2024-08-01_21-40-13.pkl"
    input_path = "kitchen_2024-08-13_17-03-52.pkl"
    # Load map
    input_path = Path(input_path)
    print("Loading:", input_path)

    demo = RobotAgent(robot, parameters, semantic_sensor, voxel_map=None)
    voxel_map = demo.voxel_map
    print("Reading from pkl file of raw observations...")
    voxel_map.read_from_pickle(input_path, num_frames=-1)

    demo.start(goal=object_to_find, visualize_map_at_start=show_intermediate_maps)

    # Run the actual procedure
    try:

        robot.switch_to_navigation_mode()
        robot.move_to_nav_posture()

        start_location = robot.get_base_pose()

        cabinet_task = np.array([0.70500136, 0.34254823, 0.85715184])
        pickup_task = np.array([0.75, -0.18, 1.57324166])
        planner = demo.planner
        res = planner.plan(start_location, cabinet_task)
        print("RES: ", res.success)

        if res.success:
            print("- Going to cabinet")
            for i, pt in enumerate(res.trajectory):
                print("-", i, pt.state)

            # Follow the planned trajectory
            robot.execute_trajectory(
                [pt.state for pt in res.trajectory],
                pos_err_threshold=pos_err_threshold,
                rot_err_threshold=rot_err_threshold,
            )
        else:
            print("[ERROR] NO PLAN COULD BE GENERATED")

        print("- Starting policy evaluation")
        robot.switch_to_manipulation_mode()
        robot.move_to_manip_posture()
        # good starting config for cabinet opening
        robot.arm_to(
            [
                0.0,  # base_x
                0.8,  # lift
                0.02,  # arm
                0.0,  # wrist yaw, pitch, roll
                -0.8,
                0.0,
            ],
            gripper=0.6,
            blocking=True,
        )
        cabinet_leader.run(display_received_images=True)
        print("- Ending policy evaluation")

        # Plan and navigate to be in front of open cabinet for pickup
        robot.switch_to_navigation_mode()
        robot.move_to_nav_posture()
        current = robot.get_base_pose()
        res = planner.plan(current, pickup_task)
        print("RES: ", res.success)

        if res.success:
            print("- Going to pickup location")
            for i, pt in enumerate(res.trajectory):
                print("-", i, pt.state)

            # Follow the planned trajectory
            robot.execute_trajectory(
                [pt.state for pt in res.trajectory],
                pos_err_threshold=pos_err_threshold,
                rot_err_threshold=rot_err_threshold,
            )
        else:
            print("[ERROR] NO PLAN COULD BE GENERATED")

        print("- Starting policy evaluation")
        robot.switch_to_manipulation_mode()
        robot.move_to_manip_posture()
        pickup_leader.run(display_received_images=True)
        print("- Ending policy evaluation")

        # Close the cabinet door with hard coded routine
        robot.arm_to(
            [
                0.0,  # base_x
                0.9,  # lift
                0.0,  # arm
                0.0,  # wrist yaw, pitch, roll
                0.0,
                0.0,
            ],
            blocking=True,
        )
        robot.switch_to_navigation_mode()
        robot.navigate_to([0.20, 0, 1.1], relative=True)

        robot.switch_to_manipulation_mode()
        robot.arm_to(
            [
                0.0,  # base_x
                0.5,  # lift
                0.03,  # arm
                0.0,  # wrist yaw, pitch, roll
                0.0,
                0.0,
            ],
            blocking=True,
        )
        robot.switch_to_navigation_mode()
        robot.navigate_to([0, 0, -1.5], relative=True)

        print("- Task finished, going home...")
        res = planner.plan(robot.get_base_pose(), start_location)
        print("RES: ", res.success)

        if res.success:
            print("- Going back to starting location")
            for i, pt in enumerate(res.trajectory):
                print("-", i, pt.state)

            # Follow the planned trajectory
            robot.execute_trajectory(
                [pt.state for pt in res.trajectory],
                pos_err_threshold=pos_err_threshold,
                rot_err_threshold=rot_err_threshold,
            )
        else:
            print("[ERROR] NO PLAN COULD BE GENERATED")

        print("Demo successful!")

    except Exception as e:
        raise (e)
    finally:
        robot.stop()


if __name__ == "__main__":
    main()
