from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from driveworld.config import load_yaml
from driveworld.data import NuScenesStaticMapRenderer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context(data_config: str, split: str):
    config = load_yaml(data_config)
    map_config = dict(config.get("static_map", {}))
    if not map_config.get("enabled", False) or not map_config.get("cache_dir"):
        raise ValueError("data config must enable static_map and define cache_dir")
    manifest = Path(config["manifest_dir"]) / f"{split}.jsonl"
    with manifest.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise RuntimeError(f"Manifest contains no records: {manifest}")
    output_dir = Path(map_config["cache_dir"])
    output = output_dir / f"{split}.packed.npy"
    return config, map_config, manifest, records, output_dir, output


def _renderer(config, map_config):
    return NuScenesStaticMapRenderer(
        config["data_root"],
        xbound=map_config.get("xbound", [-50.0, 50.0, 0.5]),
        ybound=map_config.get("ybound", [-50.0, 50.0, 0.5]),
        classes=tuple(map_config["classes"]),
    )


def _worker(data_config: str, split: str, start: int, end: int):
    config, map_config, _, records, _, output = _context(data_config, split)
    if not 0 <= start < end <= len(records):
        raise ValueError(f"Invalid cache worker interval: [{start},{end})")
    temp = output.with_suffix(output.suffix + ".tmp")
    packed = np.lib.format.open_memmap(temp, mode="r+")
    renderer = _renderer(config, map_config)
    for index in range(start, end):
        record = records[index]
        value = renderer.render(record["location"], record["map_pose"])
        packed[index] = np.packbits(
            value.astype(np.uint8, copy=False).reshape(-1), bitorder="little"
        )
    packed.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Build row-aligned bit-packed MagicDrive static-map mmap cache"
    )
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--segment-size", type=int, default=500)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=300.0,
        help="Maximum seconds allowed for one isolated cache segment",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--end", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.start is None or args.end is None:
            raise ValueError("cache worker requires --start and --end")
        _worker(args.data_config, args.split, args.start, args.end)
        return
    if args.segment_size < 1 or args.max_failures < 1 or args.worker_timeout <= 0:
        raise ValueError("segment-size/max-failures/worker-timeout must be positive")

    config, map_config, manifest, records, output_dir, output = _context(
        args.data_config, args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    metadata_path = output_dir / f"{args.split}.json"
    progress_path = output_dir / f"{args.split}.progress.json"
    manifest_sha = sha256_file(manifest)
    if output.is_file() and metadata_path.is_file() and not args.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "manifest_sha256": manifest_sha,
            "records": len(records),
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
            raise RuntimeError(f"Existing static-map cache is incompatible: {mismatched}")
        print(f"reused_static_maps={len(records)} output={output}", flush=True)
        return
    if output.is_file() or metadata_path.is_file():
        if not args.overwrite:
            raise RuntimeError(
                "Static-map cache is incomplete (data/metadata pair is missing); "
                f"use --overwrite: {output}"
            )
        output.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    next_index = 0
    if progress_path.is_file() and temp.is_file() and not args.overwrite:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("Static-map cache progress belongs to another manifest")
        next_index = int(progress["next_index"])
    else:
        progress_path.unlink(missing_ok=True)
        packed = np.lib.format.open_memmap(
            temp, mode="w+", dtype=np.uint8, shape=(len(records), 40000)
        )
        packed.flush()
        del packed

    failures = 0
    while next_index < len(records):
        end = min(next_index + args.segment_size, len(records))
        command = [
            sys.executable,
            "-m",
            "scripts.cache_static_maps",
            "--data-config",
            args.data_config,
            "--split",
            args.split,
            "--worker",
            "--start",
            str(next_index),
            "--end",
            str(end),
        ]
        try:
            worker_environment = dict(os.environ)
            worker_environment.setdefault("PYTHONMALLOC", "malloc")
            worker_environment.setdefault("MALLOC_ARENA_MAX", "2")
            result = subprocess.run(
                command,
                check=False,
                timeout=args.worker_timeout,
                env=worker_environment,
            )
        except subprocess.TimeoutExpired:
            result = subprocess.CompletedProcess(command, returncode=124)
        if result.returncode == 0:
            next_index = end
            failures = 0
            progress_temp = progress_path.with_suffix(progress_path.suffix + ".tmp")
            progress_temp.write_text(
                json.dumps(
                    {"manifest_sha256": manifest_sha, "next_index": next_index},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            progress_temp.replace(progress_path)
            print(f"cached_static_maps={next_index}/{len(records)}", flush=True)
            continue
        failures += 1
        if failures >= args.max_failures:
            raise RuntimeError(
                f"Static-map segment [{next_index},{end}) failed {failures} times"
            )
        print(
            f"isolated_cache_failure={result.returncode} segment={next_index}:{end} "
            f"attempt={failures}/{args.max_failures}",
            flush=True,
        )

    temp.replace(output)
    progress_path.unlink(missing_ok=True)
    renderer = _renderer(config, map_config)
    metadata = {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "records": len(records),
        "map_shape": [8, 200, 200],
        "packed_bytes_per_map": 40000,
        "bitorder": "little",
        "classes": list(renderer.classes),
        "xbound": list(renderer.xbound),
        "ybound": list(renderer.ybound),
        "data_root": str(config["data_root"]),
        "output": str(output),
        "output_bytes": output.stat().st_size,
    }
    metadata_temp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata_temp.replace(metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
