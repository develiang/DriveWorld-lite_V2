from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from driveworld.config import config_hash
from driveworld.utils import write_json

from .can_interpolator import CanBusInterpolator
from .geometry import relative_ego_features
from .nuscenes_tables import NuScenesTables


SCHEMA_VERSION = 3
EGO_FIELDS = ["x", "y", "yaw", "vx", "vy", "ax", "ay", "yaw_rate", "steering"]


@dataclass(frozen=True)
class ClipConfig:
    data_root: Path
    version: str
    camera: str
    fps: float
    history_frames: int
    future_frames: int
    window_stride_frames: int
    max_camera_error_ms: float
    resolution: tuple[int, int]
    manifest_dir: Path
    split: dict[str, list[str]]
    min_camera_availability: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClipConfig":
        split_value = value["split"]
        resolved_split: dict[str, list[str]] = {}
        aliases = {"official_train": "train", "official_val": "val"}
        official = None
        for split_name, scenes in split_value.items():
            if isinstance(scenes, str):
                if scenes not in aliases:
                    raise ValueError(f"Unknown split alias: {scenes}")
                if official is None:
                    try:
                        import importlib.util

                        package = importlib.util.find_spec("nuscenes")
                        if package is None or not package.submodule_search_locations:
                            raise ImportError("nuscenes package not found")
                        split_path = (
                            Path(next(iter(package.submodule_search_locations)))
                            / "utils"
                            / "splits.py"
                        )
                        spec = importlib.util.spec_from_file_location(
                            "_driveworld_nuscenes_splits", split_path
                        )
                        if spec is None or spec.loader is None:
                            raise ImportError(f"Cannot load {split_path}")
                        split_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(split_module)
                    except (ImportError, FileNotFoundError) as exc:
                        raise RuntimeError(
                            "nuscenes-devkit is required for official trainval split aliases"
                        ) from exc
                    official = split_module.create_splits_scenes()
                resolved_split[split_name] = list(official[aliases[scenes]])
            else:
                resolved_split[split_name] = list(scenes)
        return cls(
            data_root=Path(value["data_root"]),
            version=str(value["version"]),
            camera=str(value.get("camera", "CAM_FRONT")),
            fps=float(value["fps"]),
            history_frames=int(value["history_frames"]),
            future_frames=int(value["future_frames"]),
            window_stride_frames=int(value.get("window_stride_frames", 1)),
            max_camera_error_ms=float(value.get("max_camera_error_ms", 55.0)),
            resolution=tuple(map(int, value.get("resolution", [256, 448]))),
            manifest_dir=Path(value["manifest_dir"]),
            split=resolved_split,
            min_camera_availability=float(value.get("min_camera_availability", 0.0)),
        )


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    choose_left = np.abs(source[left] - target) <= np.abs(source[right] - target)
    indices = np.where(choose_left, left, right)
    errors = np.abs(source[indices] - target)
    return indices, errors


def build_scene_clips(
    tables: NuScenesTables,
    can: CanBusInterpolator,
    scene_name: str,
    split_name: str,
    config: ClipConfig,
) -> tuple[list[dict], dict]:
    records = tables.camera_records(scene_name, config.camera)
    camera_t = np.asarray([r["timestamp"] for r in records], dtype=np.int64)
    fallback_t, fallback_position, fallback_yaw = tables.ego_pose_arrays(records)
    total = config.history_frames + config.future_frames
    offsets = np.concatenate(
        [
            np.arange(-config.history_frames + 1, 1, dtype=np.float64),
            np.arange(1, config.future_frames + 1, dtype=np.float64),
        ]
    )
    if len(offsets) != total:
        raise AssertionError("history/future offset construction is inconsistent")
    interval_us = 1e6 / config.fps
    max_error_us = config.max_camera_error_ms * 1000.0
    clips: list[dict] = []
    rejected = {
        "outside": 0,
        "camera_error": 0,
        "duplicate_frame": 0,
        "missing_image": 0,
        "ego_invalid": 0,
    }
    accepted_errors: list[float] = []

    # Every camera frame can serve as anchor; stride controls overlap.
    for anchor_record_index in range(0, len(records), config.window_stride_frames):
        anchor_t = camera_t[anchor_record_index]
        target_t = np.rint(anchor_t + offsets * interval_us).astype(np.int64)
        if target_t[0] < camera_t[0] or target_t[-1] > camera_t[-1]:
            rejected["outside"] += 1
            continue
        image_indices, errors = _nearest_indices(camera_t, target_t)
        if np.any(errors > max_error_us):
            rejected["camera_error"] += 1
            continue
        if len(np.unique(image_indices)) != total or np.any(np.diff(image_indices) <= 0):
            rejected["duplicate_frame"] += 1
            continue
        selected = [records[int(i)] for i in image_indices]
        if any(not (config.data_root / record["filename"]).is_file() for record in selected):
            rejected["missing_image"] += 1
            continue
        aligned = can.interpolate(
            scene_name,
            target_t,
            fallback_t,
            fallback_position,
            fallback_yaw,
        )
        if not np.all(aligned.pose_valid):
            rejected["ego_invalid"] += 1
            continue
        ego = relative_ego_features(
            target_t,
            aligned.positions_world,
            aligned.yaw_world,
            aligned.steering,
            config.history_frames - 1,
        )
        valid = np.ones_like(ego, dtype=bool)
        valid[:, :8] = aligned.pose_valid[:, None]
        valid[:, 8] = aligned.steering_valid
        clip = {
            "schema_version": SCHEMA_VERSION,
            "clip_id": f"{scene_name}:{int(anchor_t)}",
            "scene_token": tables.scene_by_name[scene_name]["token"],
            "scene_name": scene_name,
            "location": tables.scene_location(scene_name),
            "split": split_name,
            "anchor_timestamp_us": int(anchor_t),
            "image_paths": [r["filename"] for r in selected],
            "image_timestamps_us": [int(r["timestamp"]) for r in selected],
            "target_timestamps_us": target_t.tolist(),
            "past_ego": ego[: config.history_frames].tolist(),
            "future_ego": ego[config.history_frames :].tolist(),
            "past_ego_valid": valid[: config.history_frames].tolist(),
            "future_ego_valid": valid[config.history_frames :].tolist(),
            "source_flags": {"pose": aligned.pose_source, "steering": "can_or_masked"},
            "max_time_error_ms": float(errors.max() / 1000.0),
            "camera_parameter": tables.magicdrive_camera_parameter(
                selected[config.history_frames - 1]
            ).tolist(),
            "camera_parameter_valid": True,
            "map_pose": tables.magicdrive_map_pose(
                selected[config.history_frames - 1]
            ).tolist(),
        }
        clips.append(clip)
        accepted_errors.extend((errors / 1000.0).tolist())

    stats = {
        "scene_name": scene_name,
        "split": split_name,
        "camera_frames": len(records),
        "available_camera_frames": sum(
            (config.data_root / record["filename"]).is_file() for record in records
        ),
        "camera_duration_s": float((camera_t[-1] - camera_t[0]) / 1e6),
        "accepted_clips": len(clips),
        "rejected": rejected,
        "camera_error_ms": {
            "mean": float(np.mean(accepted_errors)) if accepted_errors else None,
            "p95": float(np.percentile(accepted_errors, 95)) if accepted_errors else None,
            "max": float(np.max(accepted_errors)) if accepted_errors else None,
        },
    }
    return clips, stats


def build_manifests(config: ClipConfig, raw_config: dict[str, Any]) -> dict:
    tables = NuScenesTables(config.data_root, config.version, camera_filter=config.camera)
    can = CanBusInterpolator(config.data_root / "can_bus")
    if config.min_camera_availability:
        scene_names = sorted({scene for scenes in config.split.values() for scene in scenes})
        camera_frames = 0
        available_frames = 0
        for scene_name in scene_names:
            records = tables.camera_records(scene_name, config.camera)
            camera_frames += len(records)
            available_frames += sum(
                (config.data_root / record["filename"]).is_file() for record in records
            )
        availability = available_frames / max(camera_frames, 1)
        if availability < config.min_camera_availability:
            raise RuntimeError(
                "Dataset image availability is below the full-trainval gate: "
                f"{available_frames}/{camera_frames}={availability:.3%}, "
                f"required={config.min_camera_availability:.3%}"
            )
    config.manifest_dir.mkdir(parents=True, exist_ok=True)
    all_stats: list[dict] = []
    clip_ids: set[str] = set()
    split_counts: dict[str, int] = {}
    train_ego_values: list[np.ndarray] = []
    train_ego_valid: list[np.ndarray] = []
    for split_name, scene_names in config.split.items():
        split_path = config.manifest_dir / f"{split_name}.jsonl"
        temp_path = split_path.with_suffix(".jsonl.tmp")
        count = 0
        with temp_path.open("w", encoding="utf-8") as stream:
            for scene_name in scene_names:
                clips, stats = build_scene_clips(tables, can, scene_name, split_name, config)
                all_stats.append(stats)
                for clip in clips:
                    if clip["clip_id"] in clip_ids:
                        raise RuntimeError(f"Duplicate/leaked clip: {clip['clip_id']}")
                    clip_ids.add(clip["clip_id"])
                    stream.write(json.dumps(clip, separators=(",", ":")) + "\n")
                    count += 1
                    if split_name == "train":
                        train_ego_values.append(
                            np.asarray(clip["past_ego"] + clip["future_ego"], dtype=np.float64)
                        )
                        train_ego_valid.append(
                            np.asarray(
                                clip["past_ego_valid"] + clip["future_ego_valid"], dtype=bool
                            )
                        )
        temp_path.replace(split_path)
        split_counts[split_name] = count
    if not train_ego_values:
        raise RuntimeError("Train split produced no clips; cannot compute Ego normalization")
    values = np.concatenate(train_ego_values)
    valid = np.concatenate(train_ego_valid)
    mean = np.zeros(values.shape[1], dtype=np.float64)
    std = np.ones(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        selected = values[valid[:, column], column]
        if len(selected):
            mean[column] = selected.mean()
            std[column] = max(selected.std(), 1e-6)
    ego_stats = {
        "fields": EGO_FIELDS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "source_split": "train",
        "config_hash": config_hash(raw_config),
    }
    write_json(config.manifest_dir / "ego_stats.json", ego_stats)
    report = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash(raw_config),
        "config": raw_config,
        "split_counts": split_counts,
        "total_clips": sum(split_counts.values()),
        "ego_stats": ego_stats,
        "scenes": all_stats,
    }
    write_json(config.manifest_dir / "build_report.json", report)
    return report
