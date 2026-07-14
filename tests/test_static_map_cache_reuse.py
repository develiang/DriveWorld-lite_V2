import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from driveworld.data.nuscenes_static_map import MAGICDRIVE_MAP_CLASSES
from scripts.cache_static_maps import (
    build_manifest_index,
    main as cache_static_maps_main,
    reuse_static_map_rows,
)


def _record(clip_id: str, x: float) -> dict:
    return {
        "clip_id": clip_id,
        "location": "test-map",
        "map_pose": [x, 2.0, 0.25],
    }


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_reuse_static_map_rows_copies_only_exact_render_inputs(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    source_manifest = tmp_path / "source-manifests" / "train.jsonl"
    source_records = [_record("a", 1.0), _record("b", 3.0), _record("c", 4.0)]
    _write_manifest(source_manifest, source_records)

    source_cache = tmp_path / "source-cache"
    source_cache.mkdir()
    source_values = np.stack(
        [np.full(40000, value, dtype=np.uint8) for value in (11, 22, 33)]
    )
    np.save(source_cache / "train.packed.npy", source_values, allow_pickle=False)
    contract = {
        "classes": list(MAGICDRIVE_MAP_CLASSES),
        "xbound": [-50.0, 50.0, 0.5],
        "ybound": [-50.0, 50.0, 0.5],
    }
    source_metadata = {
        "manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "records": len(source_records),
        "map_shape": [8, 200, 200],
        "packed_bytes_per_map": 40000,
        "bitorder": "little",
        "data_root": str(data_root.resolve()),
        **contract,
    }
    (source_cache / "train.json").write_text(
        json.dumps(source_metadata), encoding="utf-8"
    )
    source_config = {
        "data_root": str(data_root),
        "manifest_dir": str(source_manifest.parent),
        "static_map": {
            "enabled": True,
            "cache_dir": str(source_cache),
            **contract,
        },
    }
    source_config_path = tmp_path / "source.yaml"
    source_config_path.write_text(yaml.safe_dump(source_config), encoding="utf-8")

    target_manifest = tmp_path / "target-manifests" / "train.jsonl"
    target_records = [
        _record("b", 3.0),
        _record("a", 1.5),  # Same clip id, different pose: it must be rendered.
        _record("d", 5.0),
    ]
    _write_manifest(target_manifest, target_records)
    target_offsets = tmp_path / "target-cache" / "train.offsets.npy"
    target_offsets.parent.mkdir()
    build_manifest_index(target_manifest, target_offsets)
    target_temp = target_offsets.parent / "train.packed.npy.tmp"
    target_values = np.lib.format.open_memmap(
        target_temp, mode="w+", dtype=np.uint8, shape=(len(target_records), 40000)
    )
    target_values.flush()
    del target_values
    target_config = {"data_root": str(data_root)}
    target_map_config = {"enabled": True, "cache_dir": "unused", **contract}

    pending, info = reuse_static_map_rows(
        source_data_config=str(source_config_path),
        split="train",
        target_config=target_config,
        target_map_config=target_map_config,
        target_manifest=target_manifest,
        target_offsets=target_offsets,
        target_temp=target_temp,
    )

    actual = np.load(target_temp, mmap_mode="r", allow_pickle=False)
    assert pending.tolist() == [1, 2]
    assert info["reused_records"] == 1
    assert np.array_equal(actual[0], source_values[1])
    assert not np.asarray(actual[1:]).any()

    # Exercise the public CLI's zero-render path with a reordered target manifest.
    complete_manifest = tmp_path / "complete-manifests" / "train.jsonl"
    _write_manifest(complete_manifest, [source_records[1], source_records[0]])
    complete_cache = tmp_path / "complete-cache"
    complete_config = {
        "data_root": str(data_root),
        "manifest_dir": str(complete_manifest.parent),
        "static_map": {
            "enabled": True,
            "cache_dir": str(complete_cache),
            **contract,
        },
    }
    complete_config_path = tmp_path / "complete.yaml"
    complete_config_path.write_text(yaml.safe_dump(complete_config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_static_maps",
            "--data-config",
            str(complete_config_path),
            "--split",
            "train",
            "--reuse-data-config",
            str(source_config_path),
        ],
    )

    cache_static_maps_main()

    complete = np.load(
        complete_cache / "train.packed.npy", mmap_mode="r", allow_pickle=False
    )
    complete_metadata = json.loads(
        (complete_cache / "train.json").read_text(encoding="utf-8")
    )
    assert np.array_equal(complete[0], source_values[1])
    assert np.array_equal(complete[1], source_values[0])
    assert complete_metadata["reused_records"] == 2
    assert complete_metadata["rendered_records"] == 0
