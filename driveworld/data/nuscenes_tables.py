from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator

from .geometry import (
    magicdrive_camera_parameter,
    pose_matrix,
    quaternion_to_yaw,
    sensor_to_ego_matrix,
)


def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Iterator[dict]:
    """Stream a large top-level JSON array without loading multi-GB nuScenes tables."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    with path.open(encoding="utf-8") as stream:
        eof = False
        while not eof:
            chunk = stream.read(chunk_size)
            eof = not chunk
            buffer = buffer[position:] + chunk
            position = 0
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"Expected JSON array in {path}")
                    position += 1
                    started = True
                    continue
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                if position >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"Expected objects in {path}")
                yield value
                position = end
        if buffer[position:].strip() not in {"", "]"}:
            raise ValueError(f"Truncated JSON array: {path}")

class NuScenesTables:
    """Small dependency-free reader for the nuScenes JSON tables used by this project."""

    def __init__(self, data_root: str | Path, version: str, camera_filter: str = "CAM_FRONT"):
        self.data_root = Path(data_root)
        self.version = version
        self.table_root = self.data_root / version
        required = ["scene", "sample", "sample_data", "ego_pose"]
        for name in required:
            if not (self.table_root / f"{name}.json").exists():
                raise FileNotFoundError(self.table_root / f"{name}.json")
        self.scenes = self._read("scene")
        self.logs = self._index("log")
        self.samples = self._index("sample")
        self.calibrated_sensors = self._index("calibrated_sensor")
        self.sensors = self._index("sensor")
        self.channel_by_calibrated_sensor = {
            token: self.sensors[record["sensor_token"]]["channel"]
            for token, record in self.calibrated_sensors.items()
        }
        camera_calibrations = {
            token
            for token, channel in self.channel_by_calibrated_sensor.items()
            if channel == camera_filter
        }
        lidar_calibrations = {
            token
            for token, channel in self.channel_by_calibrated_sensor.items()
            if channel == "LIDAR_TOP"
        }
        self.sample_data = {
            record["token"]: record
            for record in self._iter("sample_data")
            if record["calibrated_sensor_token"] in camera_calibrations
            or record["calibrated_sensor_token"] in lidar_calibrations
        }
        ego_pose_tokens = {record["ego_pose_token"] for record in self.sample_data.values()}
        self.ego_poses = {
            record["token"]: record
            for record in self._iter("ego_pose")
            if record["token"] in ego_pose_tokens
        }
        self.sample_data_by_sample: dict[str, list[dict]] = {}
        for record in self.sample_data.values():
            self.sample_data_by_sample.setdefault(record["sample_token"], []).append(record)
        self.scene_by_name = {scene["name"]: scene for scene in self.scenes}

    def scene_location(self, scene_name: str) -> str:
        scene = self.scene_by_name[scene_name]
        return str(self.logs[scene["log_token"]]["location"])

    def _read(self, name: str) -> list[dict]:
        return json.loads((self.table_root / f"{name}.json").read_text(encoding="utf-8"))

    def _iter(self, name: str):
        path = self.table_root / f"{name}.json"
        if path.stat().st_size < 128 * 1024 * 1024:
            yield from self._read(name)
        else:
            yield from iter_json_array(path)

    def _index(self, name: str) -> dict[str, dict]:
        return {record["token"]: record for record in self._read(name)}

    def camera_records(self, scene_name: str, camera: str) -> list[dict]:
        scene = self.scene_by_name[scene_name]
        first_sample = self.samples[scene["first_sample_token"]]
        candidates = [
            record
            for record in self.sample_data_by_sample[first_sample["token"]]
            if self.channel_by_calibrated_sensor[record["calibrated_sensor_token"]] == camera
            and record["is_key_frame"]
        ]
        if len(candidates) != 1:
            raise KeyError(f"Expected one {camera} keyframe in {scene_name}, got {len(candidates)}")
        token = candidates[0]["token"]
        while self.sample_data[token]["prev"]:
            token = self.sample_data[token]["prev"]
        records: list[dict] = []
        seen: set[str] = set()
        while token:
            if token in seen:
                raise RuntimeError(f"sample_data cycle detected at {token}")
            seen.add(token)
            record = self.sample_data[token]
            # Camera chains are scoped to one calibrated sensor / scene recording.
            records.append(record)
            token = record["next"]
        records.sort(key=lambda x: x["timestamp"])
        return records

    def ego_pose_arrays(self, records: list[dict]):
        import numpy as np

        timestamps = np.asarray([r["timestamp"] for r in records], dtype=np.int64)
        poses = [self.ego_poses[r["ego_pose_token"]] for r in records]
        positions = np.asarray([p["translation"][:2] for p in poses], dtype=np.float64)
        yaw = np.asarray([quaternion_to_yaw(p["rotation"]) for p in poses], dtype=np.float64)
        return timestamps, positions, yaw

    def magicdrive_camera_parameter(self, camera_record: dict):
        camera_calibration = self.calibrated_sensors[camera_record["calibrated_sensor_token"]]
        sample_token = camera_record["sample_token"]
        lidar_records = [
            record
            for record in self.sample_data_by_sample.get(sample_token, [])
            if self.channel_by_calibrated_sensor[record["calibrated_sensor_token"]]
            == "LIDAR_TOP"
            and record["is_key_frame"]
        ]
        if len(lidar_records) != 1:
            raise KeyError(
                f"Expected one LIDAR_TOP keyframe for sample {sample_token}, "
                f"got {len(lidar_records)}"
            )
        lidar_record = lidar_records[0]
        lidar_calibration = self.calibrated_sensors[lidar_record["calibrated_sensor_token"]]
        return magicdrive_camera_parameter(camera_calibration, lidar_calibration)

    def magicdrive_map_pose(self, camera_record: dict):
        """Return the lidar-frame global x/y/yaw used by LoadBEVSegmentation.

        For camera sweeps we use their timestamp-aligned ego pose and the scene's
        calibrated LIDAR_TOP transform.  This mirrors MagicDrive's
        ``lidar2global = ego2global @ lidar2ego`` map crop contract.
        """
        sample_token = camera_record["sample_token"]
        lidar_records = [
            record
            for record in self.sample_data_by_sample.get(sample_token, [])
            if self.channel_by_calibrated_sensor[record["calibrated_sensor_token"]]
            == "LIDAR_TOP"
            and record["is_key_frame"]
        ]
        if len(lidar_records) != 1:
            raise KeyError(
                f"Expected one LIDAR_TOP keyframe for sample {sample_token}, "
                f"got {len(lidar_records)}"
            )
        lidar_calibration = self.calibrated_sensors[
            lidar_records[0]["calibrated_sensor_token"]
        ]
        ego_pose = self.ego_poses[camera_record["ego_pose_token"]]
        lidar2global = pose_matrix(ego_pose) @ sensor_to_ego_matrix(lidar_calibration)
        direction = lidar2global[:3, :3] @ [1.0, 0.0, 0.0]
        import numpy as np

        return np.asarray(
            [
                lidar2global[0, 3],
                lidar2global[1, 3],
                np.arctan2(direction[1], direction[0]),
            ],
            dtype=np.float64,
        )
