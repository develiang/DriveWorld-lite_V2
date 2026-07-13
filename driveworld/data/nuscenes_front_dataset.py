from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .nuscenes_static_map import NuScenesStaticMapRenderer


class NuScenesFrontDataset:
    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        resolution: tuple[int, int] = (256, 448),
        return_numpy: bool = False,
        normalize_ego: bool = True,
        static_map: dict | None = None,
    ):
        self.manifest = Path(manifest)
        self.data_root = Path(data_root)
        self.height, self.width = resolution
        self.return_numpy = return_numpy
        self.normalize_ego_inputs = normalize_ego
        self.static_map_config = dict(static_map or {})
        self.load_static_map = bool(self.static_map_config.get("enabled", False))
        self._static_map_renderer = None
        self._static_map_cache = None
        self._static_map_cache_path = None
        with self.manifest.open(encoding="utf-8") as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        if self.load_static_map and self.static_map_config.get("cache_dir"):
            cache_dir = Path(self.static_map_config["cache_dir"])
            metadata_path = cache_dir / f"{self.manifest.stem}.json"
            cache_path = cache_dir / f"{self.manifest.stem}.packed.npy"
            if metadata_path.is_file() and cache_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
                expected = {
                    "manifest_sha256": digest,
                    "records": len(self.records),
                    "map_shape": [8, 200, 200],
                    "packed_bytes_per_map": 40000,
                    "bitorder": "little",
                }
                mismatched = {
                    key: {"expected": value, "cache": metadata.get(key)}
                    for key, value in expected.items()
                    if metadata.get(key) != value
                }
                if mismatched:
                    raise RuntimeError(f"Static-map cache metadata mismatch: {mismatched}")
                self._static_map_cache_path = cache_path
            elif self.static_map_config.get("require_cache", False):
                raise FileNotFoundError(
                    f"Required static-map cache is absent: {metadata_path} / {cache_path}"
                )
        stats_path = self.manifest.parent / "ego_stats.json"
        if normalize_ego and not stats_path.exists():
            raise FileNotFoundError(f"Missing train-derived Ego stats: {stats_path}")
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.ego_mean = np.asarray(stats["mean"], dtype=np.float32)
            self.ego_std = np.asarray(stats["std"], dtype=np.float32)
        else:
            self.ego_mean = np.zeros(9, dtype=np.float32)
            self.ego_std = np.ones(9, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def _load_static_map(self, record: dict, index: int) -> np.ndarray:
        if self._static_map_cache_path is not None:
            if self._static_map_cache is None:
                self._static_map_cache = np.load(
                    self._static_map_cache_path, mmap_mode="r", allow_pickle=False
                )
                if self._static_map_cache.shape != (len(self.records), 40000):
                    raise RuntimeError(
                        f"Unexpected static-map cache shape: {self._static_map_cache.shape}"
                    )
            packed = np.asarray(self._static_map_cache[index])
            return np.unpackbits(
                packed, bitorder="little", count=8 * 200 * 200
            ).reshape(8, 200, 200).astype(np.float32)
        missing = {"location", "map_pose"} - set(record)
        if missing:
            raise RuntimeError(
                f"Manifest schema lacks {sorted(missing)}; rebuild it with build_front_clips"
            )
        if self._static_map_renderer is None:
            self._static_map_renderer = NuScenesStaticMapRenderer(
                self.data_root,
                xbound=self.static_map_config.get("xbound", [-50.0, 50.0, 0.5]),
                ybound=self.static_map_config.get("ybound", [-50.0, 50.0, 0.5]),
                classes=tuple(
                    self.static_map_config.get(
                        "classes",
                        (
                            "drivable_area",
                            "ped_crossing",
                            "walkway",
                            "stop_line",
                            "carpark_area",
                            "road_divider",
                            "lane_divider",
                            "road_block",
                        ),
                    )
                ),
            )
        return self._static_map_renderer.render(record["location"], record["map_pose"])

    def normalize_ego(self, value: np.ndarray, valid: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(value, dtype=np.float32) - self.ego_mean) / self.ego_std
        return np.where(valid, normalized, 0.0).astype(np.float32)

    def _load_image(self, relative_path: str) -> np.ndarray:
        with Image.open(self.data_root / relative_path) as image:
            image = image.convert("RGB")
            source_w, source_h = image.size
            scale = max(self.width / source_w, self.height / source_h)
            resized = image.resize(
                (round(source_w * scale), round(source_h * scale)), Image.Resampling.BICUBIC
            )
            left = (resized.width - self.width) // 2
            top = (resized.height - self.height) // 2
            image = resized.crop((left, top, left + self.width, top + self.height))
            value = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return np.transpose(value, (2, 0, 1))

    def __getitem__(self, index: int):
        record = self.records[index]
        rgb = np.stack([self._load_image(path) for path in record["image_paths"]])
        history = len(record["past_ego"])
        past_raw = np.asarray(record["past_ego"], dtype=np.float32)
        future_raw = np.asarray(record["future_ego"], dtype=np.float32)
        past_valid = np.asarray(record["past_ego_valid"], dtype=bool)
        future_valid = np.asarray(record["future_ego_valid"], dtype=bool)
        batch = {
            "past_rgb": rgb[:history],
            "future_rgb": rgb[history:],
            "past_ego": self.normalize_ego(past_raw, past_valid) if self.normalize_ego_inputs else past_raw,
            "future_ego": self.normalize_ego(future_raw, future_valid) if self.normalize_ego_inputs else future_raw,
            "past_ego_raw": past_raw,
            "future_ego_raw": future_raw,
            "past_ego_valid": past_valid,
            "future_ego_valid": future_valid,
            "timestamps": np.asarray(record["target_timestamps_us"], dtype=np.int64),
            "clip_id": record["clip_id"],
        }
        camera_parameter = record.get("camera_parameter")
        if camera_parameter is not None:
            batch["camera_parameters"] = np.asarray(camera_parameter, dtype=np.float32)
            batch["camera_valid"] = np.asarray(
                record.get("camera_parameter_valid", True), dtype=bool
            )
        if self.load_static_map:
            batch["static_maps"] = self._load_static_map(record, index)
        if self.return_numpy:
            return batch
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is unavailable; use return_numpy=True for data tests") from exc
        return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for key, value in batch.items()}
