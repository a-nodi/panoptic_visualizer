import numpy as np


def unzip(enumerable):
    return [list(tupled_pair) for tupled_pair in list(zip(*enumerable))]

def convert_quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion to rotation matrix.
    Args:
        quaternion: 4x1 quaternion.
    Returns:
        3x3 rotation matrix.
    """
    
    x, y, z, w = quaternion
    rotation = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    
    return rotation
