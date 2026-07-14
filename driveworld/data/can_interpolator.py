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


@dataclass(frozen=True)
class PreparedCanScene:
    pose_t: np.ndarray
    positions: np.ndarray
    yaw: np.ndarray
    steering_t: np.ndarray
    steering: np.ndarray
    pose_source: str


def _load_messages(path: Path) -> list[dict]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"CAN file must contain a list: {path}")
    return value


def _prepare_columns(
    source_t: np.ndarray,
    source_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_t = np.asarray(source_t, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    if len(source_t) < 2:
        return source_t, values
    order = np.argsort(source_t)
    source_t, values = source_t[order], values[order]
    unique = np.r_[True, np.diff(source_t) > 0]
    return source_t[unique], values[unique]


def _interp_prepared_columns(
    source_t: np.ndarray,
    source_values: np.ndarray,
    target_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_t = np.asarray(target_t, dtype=np.float64)
    if len(source_t) < 2:
        shape = (len(target_t),) + source_values.shape[1:]
        return np.zeros(shape, dtype=np.float64), np.zeros(len(target_t), dtype=bool)
    valid = (target_t >= source_t[0]) & (target_t <= source_t[-1])
    flat = source_values.reshape(len(source_values), -1)
    result = np.column_stack(
        [np.interp(target_t, source_t, flat[:, column]) for column in range(flat.shape[1])]
    )
    return result.reshape((len(target_t),) + source_values.shape[1:]), valid


def _interp_columns(
    source_t: np.ndarray,
    source_values: np.ndarray,
    target_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_t, source_values = _prepare_columns(source_t, source_values)
    return _interp_prepared_columns(source_t, source_values, target_t)


class CanBusInterpolator:
    def __init__(self, can_root: str | Path):
        self.can_root = Path(can_root)
        self._prepared_scene_name: str | None = None
        self._prepared_scene: PreparedCanScene | None = None

    def prepare_scene(
        self,
        scene_name: str,
        fallback_timestamps_us: np.ndarray,
        fallback_positions_world: np.ndarray,
        fallback_yaw_world: np.ndarray,
    ) -> PreparedCanScene:
        if scene_name == self._prepared_scene_name and self._prepared_scene is not None:
            return self._prepared_scene

        pose = _load_messages(self.can_root / f"{scene_name}_pose.json")
        if len(pose) >= 2:
            pose_t = np.asarray([x["utime"] for x in pose], dtype=np.int64)
            pose_values = np.column_stack(
                [
                    np.asarray([x["pos"][:2] for x in pose], dtype=np.float64),
                    np.unwrap([quaternion_to_yaw(x["orientation"]) for x in pose]),
                ]
            )
            pose_t, pose_values = _prepare_columns(pose_t, pose_values)
            pose_source = "can_pose"
        else:
            pose_values = np.column_stack(
                [
                    np.asarray(fallback_positions_world, dtype=np.float64),
                    np.unwrap(fallback_yaw_world),
                ]
            )
            pose_t, pose_values = _prepare_columns(fallback_timestamps_us, pose_values)
            pose_source = "ego_pose_fallback"

        steering_messages = _load_messages(
            self.can_root / f"{scene_name}_steeranglefeedback.json"
        )
        if len(steering_messages) >= 2:
            steering_t = np.asarray(
                [x["utime"] for x in steering_messages], dtype=np.int64
            )
            steering = np.asarray(
                [x["value"] for x in steering_messages], dtype=np.float64
            )[:, None]
            steering_t, steering = _prepare_columns(steering_t, steering)
        else:
            steering_t = np.empty(0, dtype=np.float64)
            steering = np.empty((0, 1), dtype=np.float64)

        prepared = PreparedCanScene(
            pose_t=pose_t,
            positions=pose_values[:, :2],
            yaw=pose_values[:, 2:3],
            steering_t=steering_t,
            steering=steering,
            pose_source=pose_source,
        )
        self._prepared_scene_name = scene_name
        self._prepared_scene = prepared
        return prepared

    @staticmethod
    def interpolate_prepared(
        prepared: PreparedCanScene,
        target_timestamps_us: np.ndarray,
    ) -> InterpolatedPose:
        positions_i, pose_valid = _interp_prepared_columns(
            prepared.pose_t, prepared.positions, target_timestamps_us
        )
        yaw_i, yaw_valid = _interp_prepared_columns(
            prepared.pose_t, prepared.yaw, target_timestamps_us
        )
        pose_valid &= yaw_valid
        steering_i, steering_valid = _interp_prepared_columns(
            prepared.steering_t, prepared.steering, target_timestamps_us
        )
        return InterpolatedPose(
            positions_world=positions_i,
            yaw_world=yaw_i[:, 0],
            steering=steering_i[:, 0],
            pose_valid=pose_valid,
            steering_valid=steering_valid,
            pose_source=prepared.pose_source,
        )

    def interpolate(
        self,
        scene_name: str,
        target_timestamps_us: np.ndarray,
        fallback_timestamps_us: np.ndarray,
        fallback_positions_world: np.ndarray,
        fallback_yaw_world: np.ndarray,
    ) -> InterpolatedPose:
        prepared = self.prepare_scene(
            scene_name,
            fallback_timestamps_us,
            fallback_positions_world,
            fallback_yaw_world,
        )
        return self.interpolate_prepared(prepared, target_timestamps_us)
