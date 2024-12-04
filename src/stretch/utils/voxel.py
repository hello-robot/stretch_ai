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

"""
    This file contains a torch implementation and helpers of a
    "voxelized pointcloud" that stores features, centroids, and counts in a sparse voxel grid
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

import stretch.utils.logger as logger

USE_TORCH_GEOMETRIC = False
if USE_TORCH_GEOMETRIC:
    try:
        from torch_geometric.nn.pool.consecutive import consecutive_cluster
        from torch_geometric.nn.pool.voxel_grid import voxel_grid
        from torch_geometric.utils import scatter
    except:
        logger.warning("torch_geometric not found, falling back to custom implementation")
        USE_TORCH_GEOMETRIC = False
if not USE_TORCH_GEOMETRIC:
    from stretch.utils.torch_geometric import consecutive_cluster, voxel_grid
    from stretch.utils.torch_scatter import scatter

from typing import Literal

import torch
from sklearn.cluster import DBSCAN
from torch import Tensor


def xyz_to_flat_index(xyz, grid_size):
    """
    Convert N x 3 tensor of XYZ coordinates to flat indices.

    Args:
    xyz (torch.Tensor): N x 3 tensor of XYZ coordinates
    grid_size (torch.Tensor or list): Size of the grid in each dimension [X, Y, Z]

    Returns:
    torch.Tensor: N tensor of flat indices
    """
    if isinstance(grid_size, list):
        grid_size = torch.tensor(grid_size)

    return xyz[:, 0] + xyz[:, 1] * grid_size[0] + xyz[:, 2] * grid_size[0] * grid_size[1]


def flat_index_to_xyz(flat_index, grid_size):
    """
    Convert flat indices to N x 3 tensor of XYZ coordinates.

    Args:
    flat_index (torch.Tensor): N tensor of flat indices
    grid_size (torch.Tensor or list): Size of the grid in each dimension [X, Y, Z]

    Returns:
    torch.Tensor: N x 3 tensor of XYZ coordinates
    """
    if isinstance(grid_size, list):
        grid_size = torch.tensor(grid_size)

    z = flat_index // (grid_size[0] * grid_size[1])
    y = (flat_index % (grid_size[0] * grid_size[1])) // grid_size[0]
    x = flat_index % grid_size[0]

    return torch.stack([x, y, z], dim=1)


def merge_features(
    idx: Tensor,
    features: Tensor,
    method: Union[str, Literal["sum", "min", "max", "mean"]] = "sum",
    grid_dimensions: Optional[List[int]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Merge features based on the given indices using the specified method.

    This function takes a tensor of indices and a tensor of features, and merges
    the features for duplicate indices according to the specified method.

    Args:
        idx (Tensor): A 1D integer tensor containing indices, possibly with duplicates.
        features (Tensor): A 2D float tensor of shape (len(idx), feature_dim) containing
                           feature vectors corresponding to each index.
        method (Literal['sum', 'min', 'max', 'mean']): The method to use for merging
                                                       features. Default is 'sum'.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - A 2D tensor of shape (num_unique_idx, feature_dim) containing the
              merged features.
            - A 1D tensor of unique indices corresponding to the merged features.

    Raises:
        ValueError: If an invalid merge method is specified or if input tensors
                    have incorrect dimensions.

    Example:
        >>> idx = torch.tensor([0, 1, 0, 2, 1])
        >>> features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        >>> unique_idx, merged_features = merge_features(idx, features, method='sum')
        >>> print(merged_features)
        tensor([[ 6.0,  8.0],
                [12.0, 14.0],
                [ 7.0,  8.0]])
        >>> print(unique_idx)
        tensor([0, 1, 2])
    """
    if idx.dim() == 2 and idx.shape[-1] == 3:
        # Convert from voxel indices
        idx = xyz_to_flat_index(idx, grid_size=grid_dimensions)
    elif idx.dim() != 1:
        raise ValueError("idx must be a 1D tensor or a N x 3 tensor; was {}".format(idx.shape))
    if features.dim() != 2 or features.size(0) != idx.size(0):
        raise ValueError("features must be a 2D tensor with shape (len(idx), feature_dim)")

    unique_idx, inverse_idx = torch.unique(idx, return_inverse=True)
    num_unique = unique_idx.size(0)
    feature_dim = features.size(1)

    if method == "sum":
        merged = torch.zeros(num_unique, feature_dim, dtype=features.dtype, device=features.device)
        merged.index_add_(0, inverse_idx, features)
    elif method == "min":
        merged = torch.full(
            (num_unique, feature_dim), float("inf"), dtype=features.dtype, device=features.device
        )
        merged = torch.min(merged.index_copy(0, inverse_idx, features), merged)
    elif method == "max":
        merged = torch.full(
            (num_unique, feature_dim), -1, dtype=features.dtype, device=features.device
        )
        merged = torch.max(merged.index_copy(0, inverse_idx, features), merged)
    elif method == "mean":
        merged = torch.zeros(num_unique, feature_dim, dtype=features.dtype, device=features.device)
        merged.index_add_(0, inverse_idx, features)
        count = torch.zeros(num_unique, dtype=torch.int, device=features.device)
        count.index_add_(0, inverse_idx, torch.ones_like(inverse_idx))
        merged /= count.unsqueeze(1)
    else:
        raise ValueError("Invalid merge method. Choose from 'sum', 'min', 'max', or 'mean'.")

    if grid_dimensions is not None:
        unique_idx = flat_index_to_xyz(unique_idx, grid_size=grid_dimensions)

    return unique_idx, merged


def project_points(points_3d, K, pose):
    if not isinstance(K, torch.Tensor):
        K = torch.Tensor(K)
    K = K.to(points_3d)
    if not isinstance(pose, torch.Tensor):
        pose = torch.Tensor(pose)
    pose = pose.to(points_3d)
    # Convert points to homogeneous coordinates
    points_3d_homogeneous = torch.hstack(
        (points_3d, torch.ones((points_3d.shape[0], 1)).to(points_3d))
    )

    # Transform points into camera coordinate system
    points_camera_homogeneous = torch.matmul(torch.linalg.inv(pose), points_3d_homogeneous.T).T
    points_camera_homogeneous = points_camera_homogeneous[:, :3]

    # Project points into image plane
    points_2d_homogeneous = torch.matmul(K, points_camera_homogeneous.T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    return points_2d


def get_depth_values(points_3d, pose):
    # Convert points to homogeneous coordinates
    if not isinstance(pose, torch.Tensor):
        pose = torch.Tensor(pose)
    pose = pose.to(points_3d)
    points_3d_homogeneous = torch.hstack(
        (points_3d, torch.ones((points_3d.shape[0], 1)).to(points_3d))
    )

    # Transform points into camera coordinate system
    points_camera_homogeneous = torch.matmul(torch.linalg.inv(pose), points_3d_homogeneous.T).T

    # Extract depth values (z-coordinates)
    depth_values = points_camera_homogeneous[:, 2]

    return depth_values


class VoxelizedPointcloud:
    _INTERNAL_TENSORS = [
        "_points",
        "_features",
        "_weights",
        "_rgb",
        "dim_mins",
        "dim_maxs",
        "_mins",
        "_maxs",
    ]

    _INIT_ARGS = ["voxel_size", "dim_mins", "dim_maxs", "feature_pool_method"]

    def __init__(
        self,
        voxel_size: float = 0.05,
        dim_mins: Optional[Tensor] = None,
        dim_maxs: Optional[Tensor] = None,
        feature_pool_method: str = "mean",
    ):
        """

        Args:
            voxel_size (Tensor): float, voxel size in each dim
            dim_mins (Tensor): 3, tensor of minimum coords possible in voxel grid
            dim_maxs (Tensor): 3, tensor of maximum coords possible in voxel grid
            feature_pool_method (str, optional): How to pool features within a voxel. One of 'mean', 'max', 'sum'. Defaults to 'mean'.
        """

        assert (dim_mins is None) == (dim_maxs is None)
        self.dim_mins = dim_mins
        self.dim_maxs = dim_maxs
        self.voxel_size = voxel_size
        self.feature_pool_method = feature_pool_method
        assert self.feature_pool_method in [
            "mean",
            "max",
            "sum",
        ], f"Unknown feature pool method {feature_pool_method}"

        self.reset()

    def reset(self):
        """Resets internal tensors"""
        self._points, self._features, self._weights, self._rgb = None, None, None, None
        self._obs_counts = None
        self._mins = self.dim_mins
        self._maxs = self.dim_maxs
        self.obs_count = 1

    def remove(
        self,
        bounds: Optional[np.ndarray] = None,
        point: Optional[np.ndarray] = None,
        radius: Optional[float] = None,
        min_height: Optional[float] = None,
        min_bound_z: Optional[float] = 0.0,
    ):
        """Deletes points within a certain radius of a point, or optionally within certain bounds."""

        if min_height is None:
            min_height = -np.inf

        if point is not None and radius is not None:
            # We will do a radius removal
            assert bounds is None, "Cannot do both radius and bounds removal"
            assert len(point) == 3 or len(point) == 2, "Point must be 2 or 3D"

            if len(point) == 2:
                dists = torch.norm(self._points[:, :2] - torch.tensor(point[:2]), dim=1)
            else:
                dists = torch.norm(self._points - torch.tensor(point), dim=1)
            radius_mask = dists > radius
            height_ok = self._points[:, 2] < min_height
            mask = radius_mask | height_ok
            self._points = self._points[mask]
            if self._features is not None:
                self._features = self._features[mask]
            if self._weights is not None:
                self._weights = self._weights[mask]
            self._rgb = self._rgb[mask]

        elif bounds is not None:
            # update bounds with min z threshold
            bounds[2, 0] = max(min_bound_z, bounds[2, 0])
            if not isinstance(bounds, torch.Tensor):
                _bounds = torch.tensor(bounds)
            else:
                _bounds = bounds
            assert len(_bounds.flatten()) == 6, "Bounds must be 6D"
            mask = torch.all(self._points > _bounds[:, 0], dim=1) & torch.all(
                self._points < _bounds[:, 1], dim=1
            )
            self._points = self._points[~mask]
            if self._features is not None:
                self._features = self._features[~mask]
            if self._weights is not None:
                self._weights = self._weights[~mask]
            self._rgb = self._rgb[~mask]
        else:
            raise ValueError("Must specify either bounds or both point and radius to remove points")

    def clear_points(self, depth, intrinsics, pose, depth_is_valid=None, min_samples_clear=None):
        if self._points is not None:
            xys = project_points(self._points.detach().cpu(), intrinsics, pose).int()
            xys = xys[:, [1, 0]]
            proj_depth = get_depth_values(self._points.detach().cpu(), pose)
            H, W = depth.shape

            # Some points are projected to (i, j) on image plane and i, j might be smaller than 0 or greater than image size
            # which will lead to Index Error.
            valid_xys = xys.clone()
            valid_xys[(xys[:, 0] < 0) | (xys[:, 0] >= H) | (xys[:, 1] < 0) | (xys[:, 1] >= W)] = 0
            indices = (
                (xys[:, 0] < 0)
                | (xys[:, 0] >= H)
                | (xys[:, 1] < 0)
                | (xys[:, 1] >= W)
                # the points are projected to the image frame but is blocked by some obstacles
                | (depth[valid_xys[:, 0], valid_xys[:, 1]] < (proj_depth - 0.1))
                # the points are projected to the image frame but they are behind camera
                | (depth[valid_xys[:, 0], valid_xys[:, 1]] < 0.01)
                | (proj_depth < 0.01)
                # depth is too large
                | (proj_depth > 2.5)
            )

            indices = indices.to(self._points.device)
            self._points = self._points[indices]
            if self._features is not None:
                self._features = self._features[indices]
            if self._weights is not None:
                self._weights = self._weights[indices]
            if self._rgb is not None:
                self._rgb = self._rgb[indices]
            if self._obs_counts is not None:
                self._obs_counts = self._obs_counts[indices]

            if (
                self._points is not None
                and len(self._points) > 0
                and min_samples_clear is not None
                and min_samples_clear > 0
            ):
                dbscan = DBSCAN(eps=self.voxel_size * 4, min_samples=min_samples_clear)
                cluster_vertices = torch.cat(
                    (
                        self._points.detach().cpu(),
                        self._obs_counts.detach().cpu().reshape(-1, 1) * 1000,
                    ),
                    -1,
                ).numpy()
                clusters = dbscan.fit(cluster_vertices)
                labels = clusters.labels_
                indices = labels != -1
                self._points = self._points[indices]
                if self._features is not None:
                    self._features = self._features[indices]
                if self._weights is not None:
                    self._weights = self._weights[indices]
                if self._rgb is not None:
                    self._rgb = self._rgb[indices]
                if self._obs_counts is not None:
                    self._obs_counts = self._obs_counts[indices]

    def add(
        self,
        points: Tensor,
        features: Optional[Tensor],
        rgb: Optional[Tensor],
        weights: Optional[Tensor] = None,
        min_weight_per_voxel: float = 10.0,
        obs_count: Optional[int] = None,
    ):
        """Add a feature pointcloud to the voxel grid.

        Args:
            points (Tensor): N x 3 points to add to the voxel grid
            features (Tensor): N x D features associated with each point.
                Reduction method can be set with feature_reduciton_method in init
            rgb (Tensor): N x 3 colors s associated with each point.
            weights (Optional[Tensor], optional): Weights for each point.
                Can be detection confidence, distance to camera, etc.
                Defaults to None.
        """
        if weights is None:
            weights = torch.ones_like(points[..., 0])

        if obs_count is None:
            obs_counts = torch.ones_like(weights) * self.obs_count
        else:
            obs_counts = torch.ones_like(weights) * obs_count
        self.obs_count += 1

        # Update voxel grid bounds
        # This isn't strictly necessary since the functions below can infer the bounds
        # But we might want to do this anyway to enforce that bounds are a multiple of self.voxel_size
        # And to enforce that the added points are within user-defined boundaries, if those were specified.
        pos_mins, _ = points.min(dim=0)
        pos_maxs, _ = points.max(dim=0)
        if self.dim_mins is not None:
            assert torch.all(
                self.dim_mins <= pos_mins
            ), "Got points outside of user-defined 3D bounds"
        if self.dim_maxs is not None:
            assert torch.all(
                pos_maxs <= self.dim_maxs
            ), "Got points outside of user-defined 3D bounds"

        if self._mins is None:
            self._mins, self._maxs = pos_mins, pos_maxs
            # recompute_voxels = True
        else:
            assert self._maxs is not None, "How did self._mins get set without self._maxs?"
            # recompute_voxels = torch.any(pos_mins < self._mins) or torch.any(self._maxs < pos_maxs)
            self._mins = torch.min(self._mins, pos_mins)
            self._maxs = torch.max(self._maxs, pos_maxs)

        if self._points is None:
            assert self._features is None, "How did self._points get unset while _features is set?"
            # assert self._rgbs is None, "How did self._points get unset while _rgbs is set?"
            assert self._weights is None, "How did self._points get unset while _weights is set?"
            all_points, all_features, all_weights, all_rgb = (
                points,
                features,
                weights,
                rgb,
            )
            all_obs_counts = obs_counts
        else:
            assert (self._features is None) == (features is None)
            all_points = torch.cat([self._points, points], dim=0)
            all_weights = torch.cat([self._weights, weights], dim=0)
            all_features = (
                torch.cat([self._features, features], dim=0) if (features is not None) else None
            )
            all_rgb = torch.cat([self._rgb, rgb], dim=0) if (rgb is not None) else None

            all_obs_counts = torch.cat([self._obs_counts, obs_counts], dim=0)
        # Future optimization:
        # If there are no new voxels, then we could save a bit of compute time
        # by only recomputing the voxel/cluster for the new points
        # e.g. if recompute_voxels:
        #   raise NotImplementedError
        cluster_voxel_idx, cluster_consecutive_idx, _ = voxelize(
            all_points, voxel_size=self.voxel_size, start=self._mins, end=self._maxs
        )
        (
            self._points,
            self._features,
            self._weights,
            self._rgb,
            self._obs_counts,
        ) = reduce_pointcloud(
            cluster_consecutive_idx,
            pos=all_points,
            features=all_features,
            weights=all_weights,
            rgbs=all_rgb,
            obs_counts=all_obs_counts,
            feature_reduce=self.feature_pool_method,
            min_weight_per_voxel=min_weight_per_voxel,
        )
        self._obs_counts = self._obs_counts.int()
        return

    def get_idxs(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns voxel index (long tensor) for each point in points

        Args:
            points (Tensor): N x 3

        Returns:
            cluster_voxel_idx (Tensor): The voxel grid index (long tensor) for each point in points
            cluster_consecutive_idx (Tensor): Voxel grid reindexed to be consecutive (packed)
        """
        (
            cluster_voxel_idx,
            cluster_consecutive_idx,
            _,
        ) = voxelize(points, self.voxel_size, start=self._mins, end=self._maxs)
        return cluster_voxel_idx, cluster_consecutive_idx

    def get_voxel_idx(self, points: Tensor) -> Tensor:
        """Returns voxel index (long tensor) for each point in points

        Args:
            points (Tensor): N x 3

        Returns:
            Tensor: voxel index (long tensor) for each point in points
        """
        (
            cluster_voxel_idx,
            _,
        ) = self.get_idxs(points)
        return cluster_voxel_idx

    def get_consecutive_cluster_idx(self, points: Tensor) -> Tensor:
        """Returns voxel index (long tensor) for each point in points

        Args:
            points (Tensor): N x 3

        Returns:
            Tensor: voxel index (long tensor) for each point in points
        """
        (
            _,
            cluster_consecutive_idx,
        ) = self.get_idxs(points)
        return cluster_consecutive_idx

    def get_pointcloud(self) -> Tuple[Tensor, ...]:
        """Returns pointcloud (1 point per occupied voxel)

        Returns:
            points (Tensor): N x 3
            features (Tensor): N x D
            weights (Tensor): N
        """
        return self._points, self._features, self._weights, self._rgb

    @property
    def points(self) -> Tensor:
        return self._points

    @property
    def features(self) -> Tensor:
        return self._features

    @property
    def weights(self) -> Tensor:
        return self._weights

    @property
    def rgb(self) -> Tensor:
        return self._rgb

    @property
    def num_points(self) -> int:
        return len(self._points)

    def clone(self):
        """
        Deep copy of object. All internal tensors are cloned individually.

        Returns:
            new VoxelizedPointcloud object.
        """
        other = self.__class__(**{k: getattr(self, k) for k in self._INIT_ARGS})
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def to(self, device: Union[str, torch.device]):
        """

        Args:
          device: Device (as str or torch.device) for the new tensor.

        Returns:
          self
        """
        other = self.clone()
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.to(device))
        return other

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def detach(self):
        """
        Detach object. All internal tensors are detached individually.

        Returns:
            new VoxelizedPointcloud object.
        """
        other = self.__class__({k: getattr(self, k) for k in self._INIT_ARGS})
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())
        return other


def voxelize(
    pos: Tensor,
    voxel_size: float,
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, Tensor]] = None,
    end: Optional[Union[float, Tensor]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns voxel indices and packed (consecutive) indices for points

    Args:
        pos (Tensor): [N, 3] locations
        voxel_size (float): Size (resolution) of each voxel in the grid
        batch (Optional[Tensor], optional): Batch index of each point in pos. Defaults to None.
        start (Optional[Union[float, Tensor]], optional): Mins along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from min values in pos.
        end (Optional[Union[float, Tensor]], optional):  Maxes along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from max values in pos.
    Returns:
        voxel_idx (LongTensor): Idx of each point's voxel coordinate. E.g. [0, 0, 4, 3, 3, 4]
        cluster_consecutive_idx (LongTensor): Packed idx -- contiguous in cluster ID. E.g. [0, 0, 2, 1, 1, 2]
        batch_sample: See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/max_pool.html
    """
    voxel_cluster = voxel_grid(pos=pos, batch=batch, size=voxel_size, start=start, end=end)
    cluster_consecutive_idx, perm = consecutive_cluster(voxel_cluster)
    batch_sample = batch[perm] if batch is not None else None
    cluster_idx = voxel_cluster
    return cluster_idx, cluster_consecutive_idx, batch_sample


def scatter_weighted_mean(
    features: Tensor,
    weights: Tensor,
    cluster: Tensor,
    weights_cluster: Tensor,
    dim: int,
) -> Tensor:
    """_summary_

    Args:
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        cluster (LongTensor): [N] IDs of each point (clusters.max() should be <= N, or you'll OOM)
        weights_cluster (Tensor): [N,] aggregated weights of each cluster, used to normalize
        dim (int): Dimension along which to do the reduction -- should be 0

    Returns:
        Tensor: Agggregated features, weighted by weights and normalized by weights_cluster
    """
    assert dim == 0, "Dim != 0 not yet implemented"
    feature_cluster = scatter(features * weights[:, None], cluster, dim=dim, reduce="sum")
    feature_cluster = feature_cluster / weights_cluster[:, None]
    return feature_cluster


def reduce_pointcloud(
    voxel_cluster: Tensor,
    pos: Tensor,
    features: Tensor,
    weights: Optional[Tensor] = None,
    rgbs: Optional[Tensor] = None,
    obs_counts: Optional[Tensor] = None,
    feature_reduce: str = "mean",
    min_weight_per_voxel: float = 10.0,
) -> Tuple[Tensor, ...]:
    """Pools values within each voxel

    Args:
        voxel_cluster (LongTensor): [N] IDs of each point
        pos (Tensor): [N, 3] position of each point
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        rgbs (Optional[Tensor], optional): [N, 3] colors of each point. Defaults to None.
        feature_reduce (str, optional): Feature reduction method. Defaults to 'mean'.

    Raises:
        NotImplementedError: if unknown reduction method

    Returns:
        pos_cluster (Tensor): weighted average position within each voxel
        feature_cluster (Tensor): aggregated feature of each voxel
        weights_cluster (Tensor): aggregated weights of each voxel
        rgb_cluster (Tensor): colors of each voxel
    """
    if weights is None:
        weights = torch.ones_like(pos[..., 0])
    weights_cluster = scatter(weights, voxel_cluster, dim=0, reduce="sum")

    pos_cluster = scatter_weighted_mean(pos, weights, voxel_cluster, weights_cluster, dim=0)

    valid_idx = weights_cluster >= min_weight_per_voxel

    if rgbs is not None:
        rgb_cluster = scatter_weighted_mean(rgbs, weights, voxel_cluster, weights_cluster, dim=0)
        # rgb_cluster = rgb_cluster[valid_idx]
    else:
        rgb_cluster = None

    if obs_counts is not None:
        obs_count_cluster = scatter(obs_counts, voxel_cluster, dim=0, reduce="max")
    else:
        obs_count_cluster = None

    if features is None:
        return (
            pos_cluster,
            None,
            weights_cluster,
            rgb_cluster,
            obs_count_cluster,
        )

    if feature_reduce == "mean":
        feature_cluster = scatter_weighted_mean(
            features, weights, voxel_cluster, weights_cluster, dim=0
        )
    elif feature_reduce == "max":
        feature_cluster = scatter(features, voxel_cluster, dim=0, reduce="max")
    elif feature_reduce == "sum":
        feature_cluster = scatter(features * weights[:, None], voxel_cluster, dim=0, reduce="sum")
    else:
        raise NotImplementedError(f"Unknown feature reduction method {feature_reduce}")

    return (
        pos_cluster,
        feature_cluster,
        weights_cluster,
        rgb_cluster,
        obs_count_cluster,
    )


def scatter3d(
    voxel_indices: Tensor,
    weights: Tensor,
    grid_dimensions: List[int],
    method: Optional[str] = None,
    verbose: bool = False,
) -> Tensor:
    """Scatter weights into a 3d voxel grid of the appropriate size.

    Args:
        voxel_indices (LongTensor): [N, 3] indices to scatter values to.
        weights (FloatTensor): [N] values of equal size to scatter through voxel map.
        grid_dimenstions (List[int]): sizes of the resulting voxel map, should be 3d.
        verbose (bool): Print warnings if any. Defaults to False.

    Returns:
        voxels (FloatTensor): [grid_dimensions] voxel map containing combined weights."""

    assert voxel_indices.shape[0] == weights.shape[0], "weights and indices must match"
    assert len(grid_dimensions) == 3, "this is designed to work only in 3d"
    assert voxel_indices.shape[-1] == 3, "3d points expected for indices"

    if len(voxel_indices) == 0:
        return torch.zeros(*grid_dimensions, device=weights.device)

    N, F = weights.shape
    X, Y, Z = grid_dimensions

    # Compute voxel indices for each point
    # voxel_indices = (points / voxel_size).long().clamp(min=0, max=torch.tensor(grid_size) - 1)
    voxel_indices = voxel_indices.clamp(
        min=torch.zeros(3), max=torch.tensor(grid_dimensions) - 1
    ).long()

    # Reduce according to min/max/mean or none
    if method is not None and method != "any":
        if verbose:
            logger.warning(f"Scattering {N} points into {X}x{Y}x{Z} grid, method={method}")
        merge_features(voxel_indices, weights, grid_dimensions=grid_dimensions, method=method)

    # Create empty voxel grid
    voxel_grid = torch.zeros(*grid_dimensions, F, device=weights.device)

    # Scatter features into voxel grid
    voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = weights.float()
    voxel_grid.squeeze_(-1)
    return voxel_grid


def drop_smallest_weight_points(
    points: Tensor,
    voxel_size: float = 0.01,
    drop_prop: float = 0.1,
    min_points_after_drop: int = 3,
):
    voxel_pcd = VoxelizedPointcloud(
        voxel_size=voxel_size,
        dim_mins=None,
        dim_maxs=None,
        feature_pool_method="mean",
    )
    voxel_pcd.add(
        points=points,
        features=None,  # instance.point_cloud_features,
        rgb=None,  # instance.point_cloud_rgb,
    )
    orig_points = points
    points = voxel_pcd._points
    weights = voxel_pcd._weights
    assert len(points) > 0, points.shape
    weights_sorted, sort_idxs = torch.sort(weights, dim=0)
    points_sorted = points[sort_idxs]
    weights_cumsum = torch.cumsum(weights_sorted, dim=0)
    above_cutoff = weights_cumsum >= (drop_prop * weights_cumsum[-1])
    cutoff_idx = int(above_cutoff.max(dim=0).indices)
    if len(points_sorted[cutoff_idx:]) < min_points_after_drop:
        return orig_points
    # print(f"Reduced {len(orig_points)} -> {len(points)} -> {above_cutoff.sum()}")
    return points_sorted[cutoff_idx:]
