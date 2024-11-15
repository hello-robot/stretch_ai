# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import click
import cv2
import matplotlib

matplotlib.use("TkAgg")
import copy
import re

import numpy as np

from stretch.agent import RobotAgent
from stretch.agent.vlm_planner import VLMPlanner
from stretch.agent.zmq_client import HomeRobotZMQClient
from stretch.core import get_parameters
from stretch.core.interfaces import Observations
from stretch.perception import create_semantic_sensor
from stretch.utils.dummy_stretch_client import DummyStretchClient


def add_raw_obs_to_voxel_map(obs_history, voxel_map, semantic_sensor, num_frames, frame_skip):
    key_obs = []
    num_obs = len(obs_history["rgb"])
    video_frames = []

    print("converting raw data to observations...")
    for obs_id in range(num_obs):
        pose = obs_history["camera_poses"][obs_id]
        pose[2, 3] += 1.2  # for room1 and room2, room4, room5
        # pose[2,3] = pose[2,3] # for room3
        key_obs.append(
            Observations(
                rgb=obs_history["rgb"][obs_id].numpy(),
                # gps=obs_history["base_poses"][obs_id][:2].numpy(),
                gps=pose[:2, 3],
                # compass=[obs_history["base_poses"][obs_id][2].numpy()],
                compass=[np.arctan2(pose[1, 0], pose[0, 0])],
                xyz=None,
                depth=obs_history["depth"][obs_id].numpy(),
                camera_pose=pose,
                camera_K=obs_history["camera_K"][obs_id].numpy(),
            )
        )
        video_frames.append(obs_history["rgb"][obs_id].numpy())

    images_to_video(
        video_frames[: min(frame_skip * num_frames, len(video_frames))],
        "output_video.mp4",
        fps=10,
    )

    voxel_map.reset()
    key_obs = key_obs[::frame_skip]
    key_obs = key_obs[: min(num_frames, len(key_obs))]
    for idx, obs in enumerate(key_obs):
        print(f"processing frame {idx}")
        obs = semantic_sensor.predict(obs)
        voxel_map.add_obs(obs)

    return voxel_map


def images_to_video(image_list, output_path, fps=30):
    """
    Convert a list of raw rgb data into a video.
    """
    print("Generating an video for visualizing the data...")
    if not image_list:
        raise ValueError("The image list is empty")

    height, width, channels = image_list[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in image_list:
        if image.shape != (height, width, channels):
            raise ValueError("All images must have the same dimensions")
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(image)

    out.release()
    print(f"Video saved at {output_path}")


@click.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(),
    default="",
    help="Input path. If empty, run on the real robot.",
)
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--robot_ip", default="")
@click.option("--task", "-t", type=str, default="", help="Task to run with the planner.")
@click.option(
    "--config-path",
    "-c",
    type=click.Path(),
    default="app/vlm_planning/multi_crop_vlm_planner.yaml",
    help="Path to planner config.",
)
@click.option(
    "--frame",
    "-f",
    type=int,
    default=-1,
    help="number of frames to read",
)
@click.option(
    "--frame_skip",
    "-fs",
    type=int,
    default=1,
    help="number of frames to skip",
)
@click.option("--show-svm", "-s", type=bool, is_flag=True, default=False)
@click.option("--test-vlm", type=bool, is_flag=True, default=False)
@click.option("--show-instances", type=bool, is_flag=True, default=False)
@click.option("--api-key", type=str, default=None, help="your openai api key")
def main(
    input_path,
    config_path,
    test_vlm: bool = False,
    frame: int = -1,
    frame_skip: int = 1,
    show_svm: bool = False,
    show_instances: bool = False,
    api_key: str = None,
    task: str = "",
    local: bool = False,
    robot_ip: str = "",
):
    """Simple script to load a voxel map"""
    input_path = Path(input_path)
    print("Loading:", input_path)

    loaded_voxel_map = None

    print("- Load parameters")
    vlm_parameters = get_parameters(config_path)
    if not vlm_parameters.get("vlm_base_config"):
        print("invalid config file")
        return
    else:
        base_config_file = vlm_parameters.get("vlm_base_config")
        base_parameters = get_parameters(base_config_file)
        base_parameters.data.update(vlm_parameters.data)
        vlm_parameters.data = base_parameters.data
        print(vlm_parameters.data)

    if len(task) > 0:
        vlm_parameters.set("command", task)

    print("Creating semantic sensors...")
    semantic_sensor = create_semantic_sensor(parameters=vlm_parameters)

    if len(input_path) > 0:
        robot = DummyStretchClient()
    else:
        robot = HomeRobotZMQClient(robot_ip=robot_ip, local=local)

    print("Creating robot agent...")
    agent = RobotAgent(
        robot,
        vlm_parameters,
        voxel_map=loaded_voxel_map,
        semantic_sensor=semantic_sensor,
    )
    voxel_map = agent.voxel_map

    if len(input_path) > 0:
        # load from pickle
        voxel_map.read_from_pickle(input_path, num_frames=frame, perception=semantic_sensor)
    else:
        # Scan the local area to get a map
        agent.rotate_in_place()

    # get the task
    task = agent.get_command() if not task else task

    # or load from raw data
    # obs_history = pickle.load(input_path.open("rb"))
    # voxel_map = add_raw_obs_to_voxel_map(
    #     obs_history,
    #     voxel_map,
    #     semantic_sensor,
    #     num_frames=frame,
    #     frame_skip=frame_skip,
    # )
    run_vlm_planner(agent, task, show_svm, test_vlm, api_key, show_instances)


def run_vlm_planner(
    agent,
    task,
    show_svm: bool = False,
    test_vlm: bool = False,
    api_key: str = None,
    show_instances: bool = False,
):
    """
    Run the VLM planner with the given agent and task.

    Args:
        agent (RobotAgent): the robot agent to use.
        task (str): the task to run.
        show_svm (bool): whether to show the SVM.
        test_vlm (bool): whether to test the VLM planner.
        api_key (str): the OpenAI API key.
    """

    # TODO: read this from file
    x0 = np.array([0, 0, 0])
    # x0 = np.array([0, -0.5, 0])  # for room1, room4
    # x0 = np.array([-1.9, -0.8, 0])  # for room2
    # x0 = np.array([0, 0.5, 0])  # for room3, room5
    start_xyz = [x0[0], x0[1], 0]

    print("Agent loaded:", agent)
    vlm_parameters = agent.get_parameters()
    semantic_sensor = agent.get_semantic_sensor()
    robot = agent.get_robot()
    voxel_map = agent.get_voxel_map()

    # Create the VLM planner using the agent
    vlm_planner = VLMPlanner(agent, api_key=api_key)

    # Display with agent overlay
    space = agent.get_navigation_space()
    if show_svm:
        footprint = robot.get_footprint()
        print(f"{x0} valid = {space.is_valid(x0)}")
        voxel_map.show(instances=show_instances, orig=start_xyz, xyt=x0, footprint=footprint)

    if test_vlm:
        start_is_valid = space.is_valid(x0, verbose=True, debug=False)
        if not start_is_valid:
            print("you need to manually set the start pose to be valid")
            return

        print("\nFirst plan with the original map: ")
        original_plan, world_rep = vlm_planner.plan(
            current_pose=x0,
            show_plan=True,
            query=task,
            plan_with_reachable_instances=False,
            plan_with_scene_graph=False,
        )

        # loop over the plan and check feasibilities for each action
        preconditions = {}
        while len(original_plan) > 0:
            current_action = original_plan.pop(0)

            # navigation action only for now
            if "goto" not in current_action:
                continue

            # get the target object instance from the action
            crop_id = int(re.search(r"img_(\d+)", current_action).group(1))
            global_id = world_rep.object_images[crop_id].instance_id
            current_instance = voxel_map.get_instances()[global_id]

            print(f"Checking feasibility of action: {current_action}")
            motion_plan = agent.plan_to_instance_for_manipulation(
                current_instance, start=np.array(start_xyz)
            )
            feasible = motion_plan.get_success()

            if feasible:
                continue

            print(f"Action {current_action} is not feasible.")
            print("Searching over the map and replanning...")
            # loop over all instances in the map and try to find a feasible action
            # TODO: This is highly inefficient and should be replaced with scene graph
            planning_map = copy.deepcopy(voxel_map)
            planning_xyz = start_xyz
            for removed_instance in voxel_map.get_instances():

                # skip the current instance
                if removed_instance == current_instance:
                    continue

                # manually remove an instance for testing planning.
                new_map = copy.deepcopy(planning_map)
                new_map.delete_instance(
                    removed_instance, force_update=True, min_bound_z=0.05, assume_explored=True
                )

                # # visulize the deleted_instance and the new map
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(removed_instance.point_cloud.numpy())
                # o3d.visualization.draw_geometries([pcd])
                # new_map.show()

                # create a new agent for planning with the updated map
                planning_agent = RobotAgent(
                    robot,
                    vlm_parameters,
                    voxel_map=new_map,
                    semantic_sensor=semantic_sensor,
                )

                # motion planning with the new map
                motion_plan = planning_agent.plan_to_instance_for_manipulation(
                    removed_instance, start=np.array(planning_xyz)
                )
                feasible = motion_plan.get_success()
                if feasible:
                    print(
                        f"Found a feasible motion plan for action: {current_action} by removing instance {removed_instance.global_id}"
                    )
                    planning_xyz = motion_plan.get_trajectory()[-1].state
                    planning_map = new_map

                    # find img_id of the removed instance
                    for obj_im in world_rep.object_images:
                        if obj_im.instance_id == removed_instance.global_id:
                            removed_crop_id = obj_im.crop_id
                            break

                    # append preconditions
                    preconditions[current_action] = removed_crop_id
                    break

            print("\nPlan with the task with preconditions: ")
            print(preconditions)
            for action, crop_id in preconditions.items():
                task += f" Before {action}, relocate img_{crop_id} to another instance."
            vlm_planner.plan(
                current_pose=x0,
                show_plan=True,
                query=task,
                plan_with_reachable_instances=False,
                plan_with_scene_graph=False,
            )


if __name__ == "__main__":
    """run the test script."""
    main()
