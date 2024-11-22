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
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import torch
import trimesh.transformations as tra
from scipy.spatial import cKDTree
from trimesh import Trimesh
from trimesh.bounds import contains as trimesh_contains

rr.init("Stretch_robot", spawn=False)


def points_in_mesh(
    points: np.ndarray,
    mesh: Trimesh,
    base_pose: np.ndarray = None,
    visualize: bool = False,
) -> np.ndarray:
    """
    Check if points are inside a mesh.

    Parameters:
    points: Nx3 numpy array of points
    mesh: Trimesh object

    Returns:
    np.ndarray: Boolean array of length N
    """
    selected_indices = torch.arange(points.shape[0])

    # Fetch axis-aligned bounding box
    bbox = mesh.bounds.copy()  # 2x3 numpy array with each point containing x, y, z

    # Inflate the bounds by 0.3
    bbox[0] = bbox[0] - 0.15
    bbox[1] = bbox[1] + 0.15

    # Transform the bounds to camera frame
    x = base_pose[0]
    y = base_pose[1]
    theta = base_pose[2]

    # Rotate the bounds by 60 degrees
    # Rotation matrix for 45 degrees about the z-axis
    angle = theta
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )

    # Define the corners of the bounding box in the original space
    corners = np.array(
        [
            [bbox[0, 0], bbox[0, 1], bbox[0, 2]],  # min x, min y, min z
            [bbox[1, 0], bbox[0, 1], bbox[0, 2]],  # max x, min y, min z
            [bbox[0, 0], bbox[1, 1], bbox[0, 2]],  # min x, max y, min z
            [bbox[1, 0], bbox[1, 1], bbox[0, 2]],  # max x, max y, min z
            [bbox[0, 0], bbox[0, 1], bbox[1, 2]],  # min x, min y, max z
            [bbox[1, 0], bbox[0, 1], bbox[1, 2]],  # max x, min y, max z
            [bbox[0, 0], bbox[1, 1], bbox[1, 2]],  # min x, max y, max z
            [bbox[1, 0], bbox[1, 1], bbox[1, 2]],  # max x, max y, max z
        ]
    )

    # Rotate all corners around the z-axis
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # Find the new axis-aligned bounding box after rotation
    bbox = np.array([rotated_corners.min(axis=0), rotated_corners.max(axis=0)])

    bbox_quaternion = tra.quaternion_from_matrix(rotation_matrix)

    # Convert to xyzw quaternion
    bbox_quaternion = np.array(
        [bbox_quaternion[1], bbox_quaternion[2], bbox_quaternion[3], bbox_quaternion[0]]
    )

    # Translate the bounds by the base pose
    bbox = bbox + np.array([[x, y, 0], [x, y, 0]])

    if visualize:
        bbox_center = rr.components.PoseTranslation3D(bbox.mean(axis=0))
        bbox_half_size = (bbox[0] - bbox[1]) / 2

        rr.log(
            "world/robot_bounds",
            rr.Boxes3D(
                centers=[bbox_center],
                half_sizes=[bbox_half_size],
                quaternions=[bbox_quaternion],
                labels=["robot_bounds"],
                colors=[255, 255, 255, 255],
            ),
        )

    # Check if the points are within the bounds of the robot
    # original_length = len(selected_indices)
    if trimesh_contains(bbox, points[selected_indices].cpu().numpy()).any():
        # Modify the selected indices to remove points that are within the bounds of the robot
        selected_indices = selected_indices[
            ~trimesh_contains(bbox, points[selected_indices].cpu().numpy())
        ]
    if visualize:
        # rr.log("world/points_in_bounds", rr.Points(points[selected_indices
        # print(f"Removed {original_length - len(selected_indices)} points inside the robot bounds")
        pass

    return selected_indices


def numpy_to_pcd(
    xyz: np.ndarray, rgb: np.ndarray = None, max_points: int = -1
) -> o3d.geometry.PointCloud:
    """Create an open3d pointcloud from a single xyz/rgb pair"""
    xyz = xyz.reshape(-1, 3)
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
    if max_points > 0:
        idx = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[idx]
        if rgb is not None:
            rgb = rgb[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def pcd_to_numpy(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an open3d point cloud into xyz, rgb numpy arrays and return them."""
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    return xyz, rgb


def show_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray = None,
    orig: np.ndarray = None,
    R: np.ndarray = None,
    save: str = None,
    grasps: list = None,
    size: float = 0.1,
    max_points: int = 10000,
):
    """Shows the point-cloud described by np.ndarrays xyz & rgb.
    Optional origin and rotation params are for showing origin coordinate.
    Optional grasps param for showing a list of 6D poses as coordinate frames.
    size controls scale of coordinate frame's size
    """
    pcd = numpy_to_pcd(xyz, rgb, max_points=max_points)
    show_pcd(pcd, orig=orig, R=R, save=save, grasps=grasps, size=size)


def show_pcd(
    pcd: o3d.geometry.PointCloud,
    orig: np.ndarray = None,
    R: np.ndarray = None,
    save: str = None,
    grasps: list = None,
    size: float = 0.1,
):
    """Shows the point-cloud described by open3d.geometry.PointCloud pcd
    Optional origin and rotation params are for showing origin coordinate.
    Optional grasps param for showing a list of 6D poses as coordinate frames.
    """
    geoms = create_visualization_geometries(pcd=pcd, orig=orig, R=R, grasps=grasps, size=size)
    o3d.visualization.draw_geometries(geoms)

    if save is not None:
        save_geometries_as_image(geoms, output_path=save)


def create_visualization_geometries(
    pcd: Optional[o3d.geometry.PointCloud] = None,
    xyz: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    orig: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    size: Optional[float] = 1.0,
    arrow_pos: Optional[np.ndarray] = None,
    arrow_size: Optional[float] = 1.0,
    arrow_R: Optional[np.ndarray] = None,
    arrow_color: Optional[np.ndarray] = None,
    sphere_pos: Optional[np.ndarray] = None,
    sphere_size: Optional[float] = 1.0,
    sphere_color: Optional[np.ndarray] = None,
    grasps: list = None,
):
    """
    Creates the open3d geometries for a point cloud (one of xyz or pcd must be specified), as well as, optionally, some
    helpful indicators for points of interest -- an origin (orig), an arrow (including direction), a sphere, and grasp
    indicators.
    """
    assert (pcd is not None) != (xyz is not None), "One of pcd or xyz must be specified"

    if xyz is not None:
        xyz = xyz.reshape(-1, 3)

    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
        if np.any(rgb > 1):
            print("WARNING: rgb values too high! Normalizing...")
            rgb = rgb / np.max(rgb)

    if pcd is None:
        pcd = numpy_to_pcd(xyz, rgb)

    geoms = [pcd]
    if orig is not None:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=orig, size=size)
        if R is not None:
            coords = coords.rotate(R, orig)
        geoms.append(coords)

    if arrow_pos is not None:
        arrow = o3d.geometry.TriangleMesh.create_arrow()
        arrow = arrow.scale(
            arrow_size,
            center=np.zeros(
                3,
            ),
        )

        if arrow_color is not None:
            arrow = arrow.paint_uniform_color(arrow_color)

        if arrow_R is not None:
            arrow = arrow.rotate(arrow_R, center=(0, 0, 0))

        arrow = arrow.translate(arrow_pos)
        geoms.append(arrow)

    if sphere_pos is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size)

        if sphere_color is not None:
            sphere = sphere.paint_uniform_color(sphere_color)

        sphere = sphere.translate(sphere_pos)
        geoms.append(sphere)

    if grasps is not None:
        for grasp in grasps:
            coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=grasp[:3, 3]
            )
            coords = coords.rotate(grasp[:3, :3])
            geoms.append(coords)

    return geoms


def save_geometries_as_image(
    geoms: list,
    camera_extrinsic: Optional[np.ndarray] = None,
    look_at_point: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    zoom: Optional[float] = None,
    point_size: Optional[float] = None,
    near_clipping: Optional[float] = None,
    far_clipping: Optional[float] = None,
    live_visualization: bool = False,
):
    """
    Helper function to allow manipulation of the camera to get a better image of the point cloud.
    The live_visualization flag can help debug issues, by also spawning an interactable window.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for geom in geoms:
        vis.add_geometry(geom)
        vis.update_geometry(geom)

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    if camera_extrinsic is not None:
        # The extrinsic seems to have a different convention - switch from our camera to open3d's version
        camera_extrinsic_o3d = camera_extrinsic.copy()
        camera_extrinsic_o3d[:3, :3] = np.matmul(
            camera_extrinsic_o3d[:3, :3], np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        )
        camera_extrinsic_o3d[:, 3] = np.matmul(
            camera_extrinsic_o3d[:, 3],
            np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        )

        camera_params.extrinsic = camera_extrinsic_o3d
        view_control.convert_from_pinhole_camera_parameters(camera_params)

    if look_at_point is not None:
        view_control.set_lookat(look_at_point)

    if zoom is not None:
        view_control.set_zoom(zoom)

    if near_clipping is not None:
        view_control.set_constant_z_near(near_clipping)

    if far_clipping is not None:
        view_control.set_constant_z_far(far_clipping)

    render_options = vis.get_render_option()

    if point_size is not None:
        render_options.point_size = point_size

    if live_visualization:
        vis.run()

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()


def fix_opengl_image(rgb, depth, camera_params=None):
    """opengl images are weird, we need to do a couple things here"""
    rgb = np.flip(rgb, axis=0)
    depth = np.flip(depth, axis=0)
    if camera_params is not None:
        depth = np.clip(depth, camera_params["near_val"], camera_params["far_val"])
    return rgb, depth


def depth_to_z(depth, near, far):
    """convert depth to metric depth - based on near/far values from camera"""
    depth = 2.0 * depth - 1.0
    z = 2.0 * near * far / (far + near - depth * (far - near))
    return z


def sim_depth_to_world_xyz(depth, width, height, view_matrix, proj_matrix):
    """get points from a camera image rendered in opengl. designed to work with pybullet
    from here: https://github.com/bulletphysics/bullet3/issues/1924#issuecomment-1091876325
    """
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth value
    y, x = np.mgrid[-1 : 1 : 2 / height, -1 : 1 : 2 / width]
    y *= -1.0
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    # pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3:4]
    points = points[:, :3]
    # show_point_cloud(points)
    # import pdb; pdb.set_trace()
    return points


# We apply this correction to xyz when computing it in sim
# R_CORRECTION = R1 @ R2
R_CORRECTION = tra.euler_matrix(0, 0, np.pi / 2)[:3, :3]


def opengl_depth_to_xyz(depth, camera):
    """get depth from numpy using simple pinhole camera model"""
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(1, 2, 0)
    z = depth
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    # indices[..., 0] = np.flipud(indices[..., 0])
    x = (indices[:, :, 1] - camera.px) * (z / camera.fx)
    y = (indices[:, :, 0] - camera.py) * (z / camera.fy)  # * -1
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1) @ R_CORRECTION
    return xyz


def depth_to_xyz(depth, camera):
    """get depth from numpy using simple pinhole camera model"""
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(1, 2, 0)
    z = depth
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    x = (indices[:, :, 1] - camera.px) * (z / camera.fx)
    y = (indices[:, :, 0] - camera.py) * (z / camera.fy)
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1)
    return xyz


def build_matrix_of_indices(height, width):
    """Builds a [height, width, 2] numpy array containing coordinates.
    @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)


def pose_from_camera_params(camera_params):
    # look_at = camera_params["look_at"]
    look_from = camera_params["look_from"]
    up_vector = camera_params["up_vector"]
    pose = np.eye(4)
    pose[:3, 3] = look_from
    pose[:3, 2] = up_vector
    return pose


##### Depth augmentations #####
# from: https://github.com/chrisdxie/uois/blob/master/src/data_augmentation.py#L68


def rotate_point_cloud(pcd):
    """
    Rotate point-cloud wrt z-axis by a random angle, anti-clockwise=+ve
    """
    rotation_matrix = tra.euler_matrix(0, 0, 2 * np.pi * np.random.random() - np.pi)
    pcd = tra.transform_points(pcd, rotation_matrix)
    return pcd


def add_multiplicative_noise(depth_img, gamma_shape=10000, gamma_scale=0.0001):
    """Add noise to depth image.
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
    @param depth_img: a [H x W] set of depth z values
    """
    # TODO(cpaxton): I am pretty sure we do not need to copy here, but verify this isn't
    # causing problems
    # depth_img = depth_img.copy()

    # Multiplicative noise: Gamma random variable
    # This will randomly shift around points locally
    multiplicative_noise = np.random.gamma(gamma_shape, gamma_scale, size=depth_img.shape)
    # Apply this noise to the depth image
    depth_img = multiplicative_noise * depth_img
    return depth_img


def add_additive_noise_to_xyz(
    xyz_img,
    gp_rescale_factor_range=[12, 20],
    gaussian_scale_range=[0.0, 0.003],
    valid_mask=None,
):
    """Add (approximate) Gaussian Process noise to ordered point cloud
    @param xyz_img: a [H x W x 3] ordered point cloud
    """
    xyz_img = xyz_img.copy()

    H, W, C = xyz_img.shape

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    gp_rescale_factor = np.random.randint(gp_rescale_factor_range[0], gp_rescale_factor_range[1])
    gp_scale = np.random.uniform(gaussian_scale_range[0], gaussian_scale_range[1])

    small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
    additive_noise = np.random.normal(loc=0.0, scale=gp_scale, size=(small_H, small_W, C))
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    if valid_mask is not None:
        # use this to add to the image
        xyz_img[valid_mask, :] += additive_noise[valid_mask, :]
    else:
        xyz_img += additive_noise

    return xyz_img


def dropout_random_ellipses(depth_img, dropout_mean, gamma_shape=10000, gamma_scale=0.0001):
    """Randomly drop a few ellipses in the image for robustness.
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
    @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(dropout_mean)

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(np.where(depth_img > 0)).T  # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(
        nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout
    )
    dropout_centers = nonzero_pixel_indices[
        dropout_centers_indices, :
    ]  # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(gamma_shape, gamma_scale, size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(gamma_shape, gamma_scale, size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # dropout the ellipse
        # mask is always 2d even if input is not
        mask = np.zeros(depth_img.shape[:2])
        mask = cv2.ellipse(
            mask,
            tuple(center[::-1]),
            (x_radius, y_radius),
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1,
        )
        depth_img[mask == 1] = 0

    return depth_img


import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def ransac_transform(source_xyz, target_xyz, visualize=False, distance_threshold: float = 0.25):
    """
    Find the transformation between two point clouds using RANSAC.

    Parameters:
        source_xyz (np.ndarray): Source point cloud as a Nx3 numpy array
        target_xyz (np.ndarray): Target point cloud as a Nx3 numpy array
        visualize (bool): If True, visualize the registration result
        distance_threshold (float): Maximum correspondence distance

    Returns:
        np.ndarray: 4x4 transformation matrix
        float: Fitness score
        float: Inlier RMSE
        int: Number of inliers found
    """
    # Convert numpy arrays to Open3D point clouds
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_xyz)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_xyz)

    """
    # Estimate normals
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # Compute FPFH features
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
    )

    # Apply RANSAC registration
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=5000000, confidence=0.9999),
    )
    """

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # Visualize if flag is set
    if visualize:
        # Apply the transformation to the source point cloud
        source_transformed = source.transform(result.transformation)

        # Color the point clouds
        source_transformed.paint_uniform_color([1, 0, 0])  # Red for source
        target.paint_uniform_color([0, 1, 0])  # Green for target

        # Visualize the result
        o3d.visualization.draw_geometries(
            [source_transformed, target],
            window_name="RANSAC Registration Result",
            width=1200,
            height=800,
        )

    # Return the transformation matrix
    return result.transformation, result.fitness, result.inlier_rmse, len(result.correspondence_set)


def find_se3_transform(
    cloud1: Union[torch.Tensor, np.ndarray],
    cloud2: Union[torch.Tensor, np.ndarray],
    rgb1: Union[torch.Tensor, np.ndarray],
    rgb2: Union[torch.Tensor, np.ndarray],
    color_weight=0.5,
    max_iterations=50,
    tolerance=1e-5,
):
    """
    Find the SE(3) transformation between two colorized point clouds.

    Parameters:
    cloud1, cloud2: Nx3 numpy arrays containing XYZ coordinates
    rgb1, rgb2: Nx3 numpy arrays containing RGB values (0-255)
    color_weight: Weight given to color difference (0-1)
    max_iterations: Maximum number of ICP iterations
    tolerance: Convergence tolerance for transformation change

    Returns:
    R: 3x3 rotation matrix
    t: 3x1 translation vector
    """

    # Example usage:
    # cloud1 = np.random.rand(1000, 3)  # XYZ coordinates for cloud 1
    # cloud2 = np.random.rand(1000, 3)  # XYZ coordinates for cloud 2
    # rgb1 = np.random.randint(0, 256, (1000, 3))  # RGB values for cloud 1
    # rgb2 = np.random.randint(0, 256, (1000, 3))  # RGB values for cloud 2
    #
    # R, t = find_se3_transform(cloud1, cloud2, rgb1, rgb2)

    if isinstance(cloud1, torch.Tensor):
        cloud1 = cloud1.cpu().numpy()
    if isinstance(cloud2, torch.Tensor):
        cloud2 = cloud2.cpu().numpy()
    if isinstance(rgb1, torch.Tensor):
        rgb1 = rgb1.cpu().numpy()
    if isinstance(rgb2, torch.Tensor):
        rgb2 = rgb2.cpu().numpy()

    def best_fit_transform(A, B):
        """
        Calculates the least-squares best-fit transform between corresponding 3D points A->B
        """
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = centroid_B.T - np.dot(R, centroid_A.T)
        return R, t

    # Normalize RGB values
    rgb1_norm = rgb1 / 255.0
    rgb2_norm = rgb2 / 255.0

    # Combine spatial and color information
    combined1 = np.hstack((cloud1, rgb1_norm * color_weight))
    combined2 = np.hstack((cloud2, rgb2_norm * color_weight))

    # Initial transformation
    R = np.eye(3)
    t = np.zeros(3)

    for iteration in range(max_iterations):
        # Find nearest neighbors
        tree = cKDTree(combined2)
        distances, indices = tree.query(combined1, k=1)

        # Estimate transformation
        R_new, t_new = best_fit_transform(cloud1, cloud2[indices])

        # Update transformation
        t = t_new + np.dot(R_new, t)
        R = np.dot(R_new, R)

        # Apply transformation
        cloud1 = np.dot(cloud1, R_new.T) + t_new
        combined1[:, :3] = cloud1

        # Check for convergence
        if np.sum(np.abs(R_new - np.eye(3))) < tolerance and np.sum(np.abs(t_new)) < tolerance:
            break

    return R, t
