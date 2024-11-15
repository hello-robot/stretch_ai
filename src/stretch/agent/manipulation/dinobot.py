# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import glob
import sys

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

# sys.path.append("/home/hello-robot/repos/dino-vit-features")
sys.path.append("/home/cpaxton/src/dino-vit-features")
import sys

# Install this DINO repo to extract correspondences: https://github.com/ShirAmir/dino-vit-features
from correspondences import draw_correspondences, find_correspondences
from PIL import Image

from stretch.agent import HomeRobotZmqClient

# Hyperparameters for DINO correspondences extraction
# num_pairs = 8
# load_size = 224
# layer = 9
# facet = "key"
# bin = True
# thresh = 0.05
# model_type = "dino_vits8"
# stride = 4

# @markdown Choose number of points to output:
num_pairs = 10  # @param
# @markdown Choose loading size:
load_size = 224  # @param
# @markdown Choose layer of descriptor:
layer = 9  # @param
# @markdown Choose facet of descriptor:
facet = "key"  # @param
# @markdown Choose if to use a binned descriptor:
bin = True  # @param
# @markdown Choose fg / bg threshold:
thresh = 0.05  # @param
# @markdown Choose model type:
model_type = "dino_vits8"  # @param
# @markdown Choose stride:
stride = 4  # @param


# Deployment hyperparameters
ERR_THRESHOLD = 50  # A generic error between the two sets of points


def load_demo(path_to_demo_folder):
    # Load a demonstration from a folder containing a set of deltas.
    # The deltas are stored as numpy arrays.
    demo_deltas = []
    for filename in glob.glob(path_to_demo_folder):
        delta = np.load(filename)
        demo_deltas.append(delta)

    # pull out the first frame in the videos for the bottleneck images
    # rgb_bn = Image.open("ee_rgb_0.jpg")
    # depth_bn = Image.open("ee_rgb_1.jpg")
    rgb_bn = Image.open("bottleneck_rgb.png")
    depth_bn = Image.open("bottleneck_depth.png")

    return rgb_bn, depth_bn, demo_deltas


def show_correspondences(points1, points2, image1_pil, image2_pil):
    fig_1, ax1 = plt.subplots()
    ax1.axis("off")
    ax1.imshow(image1_pil)
    fig_2, ax2 = plt.subplots()
    ax2.axis("off")
    ax2.imshow(image2_pil)

    fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
    plt.show()


def find_transformation(X, Y):
    """
    Given two sets of 3d points, find the rigid transformation that aligns them.

    Args:
        X: np.array of shape (n, 3) representing the first set of 3d points
        Y: np.array of shape (n, 3) representing the second set of 3d points

    Returns:
        R: np.array of shape (3, 3) representing the rotation matrix
        t: np.array of shape (3,) representing the translation vector
    """

    # Calculate the centroids of the two sets of points
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)

    # Subtract the centroids to obtain two centered sets of points
    Xc = X - cX
    Yc = Y - cY

    # Calculate the covariance matrix
    C = np.dot(Xc.T, Yc)

    # Compute SVD
    U, S, Vt = np.linalg.svd(C)

    # Determine the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t


def extract_3d_coordinates(points, xyz):
    """
    Given a set of 2d points and a 3d point cloud, extract the 3d coordinates of the points.

    Args:
        points: np.array of shape (n, 2) representing the 2d points
        xyz: np.array of shape (h, w, 3) representing the 3d point cloud

    Returns:
        np.array of shape (n, 3) representing the 3d coordinates of the points

    """
    # Extract the depth values of the points from the 3d point cloud
    depths = []
    for point in points:
        x, y = point
        depths.append(xyz[y, x, 2])

    # Create a new array of shape (n, 3) with the 3d coordinates
    points_3d = []
    for i, point in enumerate(points):
        x, y = point
        points_3d.append([x, y, depths[i]])

    breakpoint()
    return np.array(points_3d)


def compute_error(points1: np.ndarray, points2: np.ndarray) -> float:
    """Compute the error between two sets of points.

    Args:
        points1: np.array of shape (n, 3) representing the first set of 3d points
        points2: np.array of shape (n, 3) representing the second set of 3d points

    Returns:
        float: The error between the two sets of points
    """
    return np.linalg.norm(np.array(points1) - np.array(points2))


def replay_demo(robot, demo_deltas):
    # Replay a demonstration by moving the robot according to the deltas.
    breakpoint()
    for delta in demo_deltas:
        robot.move(delta)


def dinobot_demo(robot, path_to_demo_folder):
    # RECORD DEMO:
    # Move the end-effector to the bottleneck pose and store the initial observation.

    # Record a demonstration. A demonstration is a set of deltas between the bottleneck pose and the current pose.
    rgb_bn, depth_bn, demo_deltas = load_demo(path_to_demo_folder)
    xyz_bn = robot.depth_to_xyz(depth_bn)

    # Reset the arm to the bottleneck pose relative to robot base
    while robot.running:
        error = 100000
        while error > ERR_THRESHOLD:
            # Collect observations at the current pose.
            rgb_live, depth_live = robot.get_ee_rgbd()
            xyz_live = robot.depth_to_xyz(depth_live)

            # Compute pixel correspondences between new observation and bottleneck observation.
            with torch.no_grad():
                # This function from an external library takes image paths as input. Therefore, store the paths of the images, and load them.
                points1, points2, image1_pil, image2_pil = find_correspondences(
                    rgb_bn,
                    rgb_live,
                    num_pairs,
                    load_size,
                    layer,
                    facet,
                    bin,
                    thresh,
                    model_type,
                    stride,
                )

                # Debug: visualize the correspondences
                show_correspondences(points1, points2, image1_pil, image2_pil)

            # Given the pixel coordinates of the correspondences, and their depth values,
            # project the points to 3D space.
            points1 = extract_3d_coordinates(points1, xyz_bn)
            points2 = extract_3d_coordinates(points2, xyz_live)

            # Find rigid translation and rotation that aligns the points by minimising error, using SVD.
            R, t = find_transformation(points1, points2)

            # Move robot
            robot.move(t, R)
            error = compute_error(points1, points2)

        # Once error is small enough, execute the demonstration.
        replay_demo(robot, demo_deltas)
        break


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option(
    "--path_to_demo_folder", default="", help="Path to the folder containing the demonstration"
)
def main(robot_ip, local, path_to_demo_folder):
    # Initialize robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        enable_rerun_server=False,
    )

    # Start the demo
    dinobot_demo(robot, path_to_demo_folder)


if __name__ == "__main__":
    main()
