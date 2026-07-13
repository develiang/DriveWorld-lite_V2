import json
from pathlib import Path

import numpy as np
import pytest

from driveworld.data import NuScenesFrontDataset


MANIFEST = Path("artifacts/manifests/nuscenes-mini-front-8x16-6hz/train.jsonl")


@pytest.mark.skipif(not MANIFEST.exists(), reason="mini manifest has not been built")
def test_real_mini_dataset_contract():
    dataset = NuScenesFrontDataset(MANIFEST, "data/nuscenes-mini", return_numpy=True)
    item = dataset[0]
    assert item["past_rgb"].shape == (8, 3, 256, 448)
    assert item["future_rgb"].shape == (16, 3, 256, 448)
    assert item["future_ego"].shape == (16, 9)
    assert item["future_ego_raw"].shape == (16, 9)
    assert np.isfinite(item["past_ego"]).all()
    assert -1 <= item["past_rgb"].min() <= item["past_rgb"].max() <= 1


@pytest.mark.skipif(not MANIFEST.exists(), reason="mini manifest has not been built")
def test_anchor_is_zero_and_scene_isolated():
    with MANIFEST.open() as stream:
        records = [json.loads(next(stream)) for _ in range(10)]
    for record in records:
        assert np.allclose(record["past_ego"][-1][:3], 0)
        assert all(record["scene_name"] in record["clip_id"] for _ in [0])
