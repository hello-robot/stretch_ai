import numpy as np
from scipy.spatial.transform import Rotation


def get_rotation_from_xyz(x_axis, y_axis, z_axis):
    # Use the gripper pose marker's orientation to directly control the robot's wrist yaw, pitch, and roll.
    rotation = np.zeros((3, 3))

    rotation[:, 0] = x_axis
    rotation[:, 1] = y_axis
    rotation[:, 2] = z_axis

    r = Rotation.from_matrix(rotation)
    return r
