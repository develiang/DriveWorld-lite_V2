from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import quaternion_to_yaw


@dataclass
class InterpolatedPose:
    positions_world: np.ndarray
    yaw_world: np.ndarray
    steering: np.ndarray
    pose_valid: np.ndarray
    steering_valid: np.ndarray
    pose_source: str


def _load_messages(path: Path) -> list[dict]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"CAN file must contain a list: {path}")
    return value


def _interp_columns(
    source_t: np.ndarray,
    source_values: np.ndarray,
    target_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_t = np.asarray(source_t, dtype=np.float64)
    target_t = np.asarray(target_t, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    if len(source_t) < 2:
        shape = (len(target_t),) + values.shape[1:]
        return np.zeros(shape, dtype=np.float64), np.zeros(len(target_t), dtype=bool)
    order = np.argsort(source_t)
    source_t, values = source_t[order], values[order]
    unique = np.r_[True, np.diff(source_t) > 0]
    source_t, values = source_t[unique], values[unique]
    valid = (target_t >= source_t[0]) & (target_t <= source_t[-1])
    flat = values.reshape(len(values), -1)
    result = np.column_stack(
        [np.interp(target_t, source_t, flat[:, column]) for column in range(flat.shape[1])]
    )
    return result.reshape((len(target_t),) + values.shape[1:]), valid


class CanBusInterpolator:
    def __init__(self, can_root: str | Path):
        self.can_root = Path(can_root)

    def interpolate(
        self,
        scene_name: str,
        target_timestamps_us: np.ndarray,
        fallback_timestamps_us: np.ndarray,
        fallback_positions_world: np.ndarray,
        fallback_yaw_world: np.ndarray,
    ) -> InterpolatedPose:
        pose = _load_messages(self.can_root / f"{scene_name}_pose.json")
        if len(pose) >= 2:
            pose_t = np.array([x["utime"] for x in pose], dtype=np.int64)
            positions = np.asarray([x["pos"][:2] for x in pose], dtype=np.float64)
            yaw = np.unwrap([quaternion_to_yaw(x["orientation"]) for x in pose])
            positions_i, pose_valid = _interp_columns(pose_t, positions, target_timestamps_us)
            yaw_i, yaw_valid = _interp_columns(pose_t, yaw[:, None], target_timestamps_us)
            pose_valid &= yaw_valid
            pose_source = "can_pose"
        else:
            positions_i, pose_valid = _interp_columns(
                fallback_timestamps_us, fallback_positions_world, target_timestamps_us
            )
            unwrapped = np.unwrap(fallback_yaw_world)
            yaw_i, yaw_valid = _interp_columns(
                fallback_timestamps_us, unwrapped[:, None], target_timestamps_us
            )
            pose_valid &= yaw_valid
            pose_source = "ego_pose_fallback"

        steering_messages = _load_messages(
            self.can_root / f"{scene_name}_steeranglefeedback.json"
        )
        if len(steering_messages) >= 2:
            steering_t = np.array([x["utime"] for x in steering_messages], dtype=np.int64)
            steering_values = np.array([x["value"] for x in steering_messages], dtype=np.float64)
            steering_i, steering_valid = _interp_columns(
                steering_t, steering_values[:, None], target_timestamps_us
            )
            steering_i = steering_i[:, 0]
        else:
            steering_i = np.zeros(len(target_timestamps_us), dtype=np.float64)
            steering_valid = np.zeros(len(target_timestamps_us), dtype=bool)

        return InterpolatedPose(
            positions_world=positions_i,
            yaw_world=yaw_i[:, 0],
            steering=steering_i,
            pose_valid=pose_valid,
            steering_valid=steering_valid,
            pose_source=pose_source,
        )

