from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NuScenesLatentDataset:
    """Dataset backed by frozen VAE cache; no RGB decode or VAE memory is needed."""

    def __init__(self, manifest: str | Path, cache_index: str | Path, allow_incomplete: bool = False):
        import torch  # noqa: F401 - fail early with a useful environment error.

        self.manifest = Path(manifest)
        self.cache_index = Path(cache_index)
        with self.manifest.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        with self.cache_index.open(encoding="utf-8") as stream:
            cache_records = [json.loads(line) for line in stream if line.strip()]
        cache_by_clip = {row["clip_id"]: self.cache_index.parent / row["path"] for row in cache_records}
        self.records = [record for record in records if record["clip_id"] in cache_by_clip]
        if len(self.records) != len(records) and not allow_incomplete:
            missing = len(records) - len(self.records)
            raise RuntimeError(f"Latent cache is incomplete: {missing}/{len(records)} clips missing")
        if not self.records:
            raise RuntimeError("Latent cache contains no clips from the manifest")
        self.cache_by_clip = cache_by_clip
        stats = json.loads((self.manifest.parent / "ego_stats.json").read_text(encoding="utf-8"))
        self.ego_mean = np.asarray(stats["mean"], dtype=np.float32)
        self.ego_std = np.asarray(stats["std"], dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def _normalize(self, value, valid):
        value = (np.asarray(value, dtype=np.float32) - self.ego_mean) / self.ego_std
        return np.where(valid, value, 0.0).astype(np.float32)

    def __getitem__(self, index):
        import torch

        record = self.records[index]
        cache = torch.load(self.cache_by_clip[record["clip_id"]], map_location="cpu", weights_only=True)
        future_valid = np.asarray(record["future_ego_valid"], dtype=bool)
        return {
            "past_latent": cache["past"],
            "future_latent": cache["future"],
            "future_ego": torch.from_numpy(self._normalize(record["future_ego"], future_valid)),
            "future_ego_valid": torch.from_numpy(future_valid),
            "clip_id": record["clip_id"],
        }
