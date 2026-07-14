import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from driveworld.data import NuScenesFrontDataset
from driveworld.data.nuscenes_static_map import (
    MAGICDRIVE_MAP_CLASSES,
    NuScenesStaticMapRenderer,
)
from scripts.cache_static_maps import main as cache_static_maps_main


MANIFEST = Path("artifacts/manifests/nuscenes-mini-front-8x16-6hz/train.jsonl")
PARTIAL_MANIFEST = Path(
    "artifacts/manifests/nuscenes-trainval-partial-front-8x16-6hz/val.jsonl"
)


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


@pytest.mark.skipif(not PARTIAL_MANIFEST.exists(), reason="partial trainval manifest is unavailable")
def test_v2_manifest_exposes_magicdrive_camera_parameter():
    dataset = NuScenesFrontDataset(
        PARTIAL_MANIFEST, "data/nuscenes-trainval", return_numpy=True
    )
    item = dataset[0]
    assert item["camera_parameters"].shape == (3, 7)
    assert bool(item["camera_valid"])
    assert np.isfinite(item["camera_parameters"]).all()


@pytest.mark.skipif(
    not PARTIAL_MANIFEST.exists()
    or not Path(
        "data/nuscenes-trainval/maps/expansion/singapore-onenorth.json"
    ).exists(),
    reason="partial manifest or nuScenes semantic map expansion is unavailable",
)
def test_v2_dataset_renders_exact_magicdrive_static_map_contract():
    dataset = NuScenesFrontDataset(
        PARTIAL_MANIFEST,
        "data/nuscenes-trainval",
        return_numpy=True,
        static_map={"enabled": True},
    )
    item = dataset[0]
    static_map = item["static_maps"]
    assert static_map.shape == (8, 200, 200)
    assert static_map.dtype == np.float32
    assert set(np.unique(static_map)).issubset({0.0, 1.0})
    assert static_map.sum() > 0


@pytest.mark.skipif(
    not PARTIAL_MANIFEST.exists()
    or not Path(
        "data/nuscenes-trainval/maps/expansion/singapore-onenorth.json"
    ).exists(),
    reason="partial manifest or nuScenes semantic map expansion is unavailable",
)
def test_v2_dataset_reads_bitpacked_static_map_cache(tmp_path):
    record = json.loads(PARTIAL_MANIFEST.open(encoding="utf-8").readline())
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    manifest = manifest_dir / "val.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    source_stats = PARTIAL_MANIFEST.parent / "ego_stats.json"
    (manifest_dir / "ego_stats.json").write_bytes(source_stats.read_bytes())

    renderer = NuScenesStaticMapRenderer("data/nuscenes-trainval")
    expected = renderer.render(record["location"], record["map_pose"])
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    packed = np.packbits(
        expected.astype(np.uint8).reshape(1, -1), axis=1, bitorder="little"
    )
    np.save(cache_dir / "val.packed.npy", packed, allow_pickle=False)
    metadata = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "records": 1,
        "map_shape": [8, 200, 200],
        "packed_bytes_per_map": 40000,
        "bitorder": "little",
    }
    (cache_dir / "val.json").write_text(json.dumps(metadata), encoding="utf-8")

    dataset = NuScenesFrontDataset(
        manifest,
        "data/nuscenes-trainval",
        return_numpy=True,
        static_map={"enabled": True, "cache_dir": str(cache_dir), "require_cache": True},
    )
    actual = dataset._load_static_map(dataset.records[0], 0)
    assert np.array_equal(actual, expected)


@pytest.mark.skipif(
    not PARTIAL_MANIFEST.exists()
    or not Path(
        "data/nuscenes-trainval/maps/expansion/singapore-onenorth.json"
    ).exists(),
    reason="partial manifest or nuScenes semantic map expansion is unavailable",
)
def test_static_map_cache_parallel_workers_write_disjoint_rows(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    with PARTIAL_MANIFEST.open(encoding="utf-8") as source:
        lines = [next(source) for _ in range(4)]
    (manifest_dir / "train.jsonl").write_text("".join(lines), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    config = {
        "data_root": str(Path("data/nuscenes-trainval").resolve()),
        "manifest_dir": str(manifest_dir),
        "static_map": {
            "enabled": True,
            "cache_dir": str(cache_dir),
            "xbound": [-50.0, 50.0, 0.5],
            "ybound": [-50.0, 50.0, 0.5],
            "classes": list(MAGICDRIVE_MAP_CLASSES),
        },
    }
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_static_maps",
            "--data-config",
            str(config_path),
            "--split",
            "train",
            "--segment-size",
            "1",
            "--workers",
            "2",
        ],
    )

    cache_static_maps_main()

    packed = np.load(cache_dir / "train.packed.npy", mmap_mode="r", allow_pickle=False)
    metadata = json.loads((cache_dir / "train.json").read_text(encoding="utf-8"))
    assert packed.shape == (4, 40000)
    assert np.all(np.asarray(packed).sum(axis=1) > 0)
    assert metadata["records"] == 4
