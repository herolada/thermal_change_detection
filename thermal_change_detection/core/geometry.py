"""Pose helpers. All camera poses are 4x4 T_map_cam (camera -> map) in the
optical convention: z forward, x right, y down."""
import numpy as np

# Rotation from a ROS body-convention frame (x fwd, y left, z up) to the
# optical frame: columns are the optical axes expressed in the body frame.
R_BODY_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])
T_BODY_OPTICAL = np.eye(4)
T_BODY_OPTICAL[:3, :3] = R_BODY_OPTICAL


def quat_to_rot(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform_to_matrix(translation, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(*quat_xyzw)
    T[:3, 3] = translation
    return T


def optical_axis(T_map_cam):
    return T_map_cam[:3, 2]


def axis_angle_deg(axes, axis):
    """Angle between one unit vector and an (N,3) array of unit vectors."""
    c = np.clip(axes @ axis, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def rotation_homography(K_cur, K_ref, T_map_cur, T_map_ref):
    """Homography mapping reference pixels to current pixels for points at
    infinity (ignores the translation between the two cameras)."""
    R_cur_ref = T_map_cur[:3, :3].T @ T_map_ref[:3, :3]
    H = K_cur @ R_cur_ref @ np.linalg.inv(K_ref)
    return H / H[2, 2]
