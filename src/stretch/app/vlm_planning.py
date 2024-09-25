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

import numpy as np
import open3d as o3d

from stretch.agent import RobotAgent
from stretch.agent.vlm_planner import VLMPlanner
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
    default="output.pkl",
    help="Input path with default value 'output.npy'",
)
@click.option(
    "--config-path",
    "-c",
    type=click.Path(),
    # default="default_planner.yaml",
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
# @click.option("--query", type=str, default=None)
def main(
    input_path,
    config_path,
    test_vlm: bool = False,
    frame: int = -1,
    frame_skip: int = 1,
    show_svm: bool = False,
    show_instances: bool = False,
    api_key: str = None,
    # query: str = None,
):
    """Simple script to load a voxel map"""
    input_path = Path(input_path)
    print("Loading:", input_path)

    loaded_voxel_map = None

    dummy_robot = DummyStretchClient()

    print("- Load parameters")
    parameters = get_parameters(config_path)
    parameters.set("vlm_option", "gpt4")

    print("Creating semantic sensors...")
    semantic_sensor = create_semantic_sensor(config_path=config_path)

    print("Creating robot agent...")
    agent = RobotAgent(
        dummy_robot,
        parameters,
        voxel_map=loaded_voxel_map,
        semantic_sensor=semantic_sensor,
    )
    voxel_map = agent.voxel_map
    voxel_map.read_from_pickle(input_path, num_frames=frame, perception=semantic_sensor)

    # or load from raw data
    # obs_history = pickle.load(input_path.open("rb"))
    # voxel_map = add_raw_obs_to_voxel_map(
    #     obs_history,
    #     voxel_map,
    #     semantic_sensor,
    #     num_frames=frame,
    #     frame_skip=frame_skip,
    # )

    # TODO: read this from file
    x0 = np.array([0, 0, 0])
    # x0 = np.array([0, -0.5, 0])  # for room1, room4
    # x0 = np.array([-1.9, -0.8, 0])  # for room2
    # x0 = np.array([0, 0.5, 0])  # for room3, room5
    start_xyz = [x0[0], x0[1], 0]

    print("Agent loaded:", agent)

    # Create the VLM planner using the agent
    vlm_planner = VLMPlanner(agent, api_key=api_key)

    # Display with agent overlay
    space = agent.get_navigation_space()
    if show_svm:
        footprint = dummy_robot.get_footprint()
        print(f"{x0} valid = {space.is_valid(x0)}")
        voxel_map.show(instances=show_instances, orig=start_xyz, xyt=x0, footprint=footprint)

    if test_vlm:
        start_is_valid = space.is_valid(x0, verbose=True, debug=False)
        if not start_is_valid:
            print("you need to manually set the start pose to be valid")
            return

        # manually remove an instance for testing planning. TODO: this should come from VLM.
        new_map = copy.deepcopy(voxel_map)
        pcd = o3d.geometry.PointCloud()
        removed_instance = voxel_map.get_instances()[0]  # hopefully this is a stuffed animal
        pcd.points = o3d.utility.Vector3dVector(removed_instance.point_cloud.numpy())
        o3d.visualization.draw_geometries([pcd])
        new_map.delete_instance(removed_instance, force_update=True, min_bound_z=0.05)
        new_map.show()

        # create a new agent for planning with the updated map
        planning_agent = RobotAgent(
            dummy_robot,
            parameters,
            voxel_map=new_map,
            semantic_sensor=semantic_sensor,
        )

        print("\nPlan with the original map: ")
        vlm_planner.plan(current_pose=x0, show_plan=True)
        print("\nPlan with the updated map: ")
        vlm_planner.set_agent(planning_agent)
        vlm_planner.plan(current_pose=x0, show_plan=True)


if __name__ == "__main__":
    """run the test script."""
    main()
