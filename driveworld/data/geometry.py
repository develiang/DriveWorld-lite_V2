from __future__ import annotations

import math

import numpy as np


def quaternion_to_yaw(q: list[float] | np.ndarray) -> float:
    """Return planar yaw for a nuScenes quaternion in [w, x, y, z] order."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def quaternion_to_rotation_matrix(q: list[float] | np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix for nuScenes [w,x,y,z] quaternions."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("Quaternion norm cannot be zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def sensor_to_ego_matrix(calibration: dict) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(calibration["rotation"])
    transform[:3, 3] = np.asarray(calibration["translation"], dtype=np.float64)
    return transform


def pose_matrix(pose: dict) -> np.ndarray:
    """Construct a homogeneous local-to-global matrix from a nuScenes pose."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(pose["rotation"])
    transform[:3, 3] = np.asarray(pose["translation"], dtype=np.float64)
    return transform


def magicdrive_camera_parameter(camera_calibration: dict, lidar_calibration: dict) -> np.ndarray:
    """Construct MagicDrive's [intrinsic(3x3), camera2lidar(3x4)] tensor."""
    intrinsic = np.asarray(camera_calibration["camera_intrinsic"], dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected camera intrinsic 3x3, got {intrinsic.shape}")
    camera2ego = sensor_to_ego_matrix(camera_calibration)
    lidar2ego = sensor_to_ego_matrix(lidar_calibration)
    camera2lidar = np.linalg.inv(lidar2ego) @ camera2ego
    return np.concatenate([intrinsic, camera2lidar[:3]], axis=1).astype(np.float32)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def world_to_anchor_xy(points: np.ndarray, anchor_xy: np.ndarray, anchor_yaw: float) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float64) - np.asarray(anchor_xy, dtype=np.float64)
    c, s = math.cos(anchor_yaw), math.sin(anchor_yaw)
    rotation = np.array([[c, s], [-s, c]], dtype=np.float64)
    return delta @ rotation.T


def relative_ego_features(
    timestamps_us: np.ndarray,
    positions_world: np.ndarray,
    yaw_world: np.ndarray,
    steering: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """Construct [x,y,yaw,vx,vy,ax,ay,yaw_rate,steering] in anchor frame."""
    seconds = (timestamps_us - timestamps_us[anchor_index]).astype(np.float64) / 1e6
    yaw_world = np.unwrap(np.asarray(yaw_world, dtype=np.float64))
    xy = world_to_anchor_xy(
        np.asarray(positions_world, dtype=np.float64),
        np.asarray(positions_world[anchor_index], dtype=np.float64),
        float(yaw_world[anchor_index]),
    )
    relative_yaw = np.unwrap(yaw_world - yaw_world[anchor_index])
    edge_order = 2 if len(seconds) >= 3 else 1
    velocity = np.gradient(xy, seconds, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, seconds, axis=0, edge_order=edge_order)
    yaw_rate = np.gradient(relative_yaw, seconds, edge_order=edge_order)
    features = np.column_stack(
        [xy, relative_yaw, velocity, acceleration, yaw_rate, np.asarray(steering)]
    )
    features[anchor_index, :3] = 0.0
    return features.astype(np.float32)
