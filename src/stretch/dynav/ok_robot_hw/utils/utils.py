import numpy as np

def apply_se3_transform(se3_obj, point):
    homogeneous_point = np.append(point.flatten(), 1)
    print(homogeneous_point)
    transformed_homogeneous_point = se3_obj.homogeneous.dot(homogeneous_point)
    transformed_point = transformed_homogeneous_point[:3]

    return transformed_point

def transform_joint_array(joint_array):
    n = len(joint_array)
    new_joint_array = []
    for i in range(n+3):
        if i < 2:
            new_joint_array.append(joint_array[i])
        elif i < 6:
            new_joint_array.append(joint_array[2]/4.0)
        else:
            new_joint_array.append(joint_array[i-3])
    return np.array(new_joint_array)