from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
    output_dir = Path(map_config["cache_dir"])
    output = output_dir / f"{split}.packed.npy"
    offsets = output_dir / f"{split}.offsets.npy"
    return config, map_config, manifest, output_dir, output, offsets


def build_manifest_index(manifest: Path, offsets_path: Path) -> tuple[str, int]:
    """Build byte offsets for non-empty JSONL records while hashing the manifest."""
    digest = hashlib.sha256()
    offsets: list[int] = []
    with manifest.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            digest.update(line)
            if line.strip():
                offsets.append(offset)
    if not offsets:
        raise RuntimeError(f"Manifest contains no records: {manifest}")
    values = np.asarray(offsets, dtype=np.uint64)
    temp = offsets_path.with_suffix(offsets_path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temp.replace(offsets_path)
    return digest.hexdigest(), len(values)


def iter_manifest_range(
    manifest: Path,
    offsets_path: Path,
    start: int,
    end: int,
):
    """Yield an indexed slice without parsing or retaining the rest of the JSONL."""
    offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    if offsets.ndim != 1 or offsets.dtype != np.dtype(np.uint64):
        raise RuntimeError(f"Invalid manifest offset index: {offsets_path}")
    if not 0 <= start < end <= len(offsets):
        raise ValueError(f"Invalid cache worker interval: [{start},{end})")
    with manifest.open("rb") as stream:
        for index in range(start, end):
            stream.seek(int(offsets[index]))
            line = stream.readline()
            if not line.strip():
                raise RuntimeError(f"Manifest index points to an empty line: {index}")
            yield index, json.loads(line)


def iter_manifest_indices(
    manifest: Path,
    offsets_path: Path,
    indices,
):
    """Yield arbitrary indexed records without reading unrelated JSONL rows."""
    offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    if offsets.ndim != 1 or offsets.dtype != np.dtype(np.uint64):
        raise RuntimeError(f"Invalid manifest offset index: {offsets_path}")
    with manifest.open("rb") as stream:
        for value in indices:
            index = int(value)
            if not 0 <= index < len(offsets):
                raise ValueError(f"Invalid manifest record index: {index}")
            stream.seek(int(offsets[index]))
            line = stream.readline()
            if not line.strip():
                raise RuntimeError(f"Manifest index points to an empty line: {index}")
            yield index, json.loads(line)


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temp.replace(path)


def _reuse_key(record: dict):
    """Only reuse rows whose complete static-map input is exactly identical."""
    pose = record.get("map_pose")
    if not isinstance(pose, list) or len(pose) != 3:
        raise RuntimeError(
            f"Manifest record {record.get('clip_id', '?')} lacks a valid map_pose"
        )
    return (
        record.get("clip_id"),
        record.get("location"),
        float(pose[0]),
        float(pose[1]),
        float(pose[2]),
    )


def _map_contract(config: dict, map_config: dict) -> dict:
    return {
        "classes": list(map_config["classes"]),
        "xbound": [float(value) for value in map_config.get("xbound", [-50, 50, 0.5])],
        "ybound": [float(value) for value in map_config.get("ybound", [-50, 50, 0.5])],
        "data_root": str(Path(config["data_root"]).resolve()),
    }


def reuse_static_map_rows(
    *,
    source_data_config: str,
    split: str,
    target_config: dict,
    target_map_config: dict,
    target_manifest: Path,
    target_offsets: Path,
    target_temp: Path,
    skip_before: int = 0,
) -> tuple[np.ndarray, dict]:
    """Copy exact source-cache matches and return target rows still needing render."""
    (
        source_config,
        source_map_config,
        source_manifest,
        _,
        source_output,
        _,
    ) = _context(source_data_config, split)
    source_metadata_path = source_output.parent / f"{split}.json"
    if source_output.resolve() == target_temp.with_suffix("").resolve():
        raise ValueError("Static-map cache cannot reuse itself")
    if not source_output.is_file() or not source_metadata_path.is_file():
        raise FileNotFoundError(
            "Reusable static-map cache is incomplete: "
            f"{source_output} / {source_metadata_path}"
        )
    source_contract = _map_contract(source_config, source_map_config)
    target_contract = _map_contract(target_config, target_map_config)
    if source_contract != target_contract:
        raise RuntimeError(
            "Reusable static-map cache has a different rendering contract: "
            f"source={source_contract}, target={target_contract}"
        )

    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_manifest_sha = sha256_file(source_manifest)
    expected_metadata = {
        "manifest_sha256": source_manifest_sha,
        "map_shape": [8, 200, 200],
        "packed_bytes_per_map": 40000,
        "bitorder": "little",
        "classes": source_contract["classes"],
        "xbound": source_contract["xbound"],
        "ybound": source_contract["ybound"],
    }
    mismatched = {
        key: {"expected": value, "cache": source_metadata.get(key)}
        for key, value in expected_metadata.items()
        if source_metadata.get(key) != value
    }
    cached_data_root = source_metadata.get("data_root")
    if cached_data_root is None or str(Path(cached_data_root).resolve()) != source_contract["data_root"]:
        mismatched["data_root"] = {
            "expected": source_contract["data_root"],
            "cache": cached_data_root,
        }
    if mismatched:
        raise RuntimeError(f"Reusable static-map cache is incompatible: {mismatched}")

    source_rows: dict[tuple, int] = {}
    with source_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            key = _reuse_key(json.loads(line))
            if key in source_rows:
                raise RuntimeError(f"Duplicate reusable static-map record: {key[0]}")
            source_rows[key] = len(source_rows)
    expected_records = int(source_metadata.get("records", -1))
    if expected_records != len(source_rows):
        raise RuntimeError(
            "Reusable static-map cache record count mismatch: "
            f"metadata={expected_records}, manifest={len(source_rows)}"
        )

    source_packed = np.load(source_output, mmap_mode="r", allow_pickle=False)
    if source_packed.shape != (len(source_rows), 40000) or source_packed.dtype != np.uint8:
        raise RuntimeError(
            f"Unexpected reusable static-map cache array: "
            f"shape={source_packed.shape}, dtype={source_packed.dtype}"
        )
    target_packed = np.lib.format.open_memmap(target_temp, mode="r+")
    pending: list[int] = []
    target_batch: list[int] = []
    source_batch: list[int] = []
    reused = 0

    def flush_batch() -> None:
        if not target_batch:
            return
        target_packed[np.asarray(target_batch, dtype=np.intp)] = source_packed[
            np.asarray(source_batch, dtype=np.intp)
        ]
        target_batch.clear()
        source_batch.clear()

    target_count = len(np.load(target_offsets, mmap_mode="r", allow_pickle=False))
    if not 0 <= skip_before <= target_count:
        raise ValueError(f"Invalid completed cache prefix: {skip_before}")
    if skip_before < target_count:
        for index, record in iter_manifest_range(
            target_manifest, target_offsets, skip_before, target_count
        ):
            source_index = source_rows.get(_reuse_key(record))
            if source_index is None:
                pending.append(index)
                continue
            target_batch.append(index)
            source_batch.append(source_index)
            reused += 1
            if len(target_batch) >= 256:
                flush_batch()
    flush_batch()
    target_packed.flush()
    return np.asarray(pending, dtype=np.uint64), {
        "reused_records": reused,
        "reuse_source": str(source_output),
        "reuse_source_manifest_sha256": source_manifest_sha,
    }


def _renderer(config, map_config):
    return NuScenesStaticMapRenderer(
        config["data_root"],
        xbound=map_config.get("xbound", [-50.0, 50.0, 0.5]),
        ybound=map_config.get("ybound", [-50.0, 50.0, 0.5]),
        classes=tuple(map_config["classes"]),
    )


def _worker(
    data_config: str,
    split: str,
    start: int,
    end: int,
    indices_path: str | None = None,
):
    import time
    config, map_config, manifest, _, output, offsets = _context(data_config, split)
    temp = output.with_suffix(output.suffix + ".tmp")
    packed = np.lib.format.open_memmap(temp, mode="r+")
    renderer = _renderer(config, map_config)
    t_start = time.time()
    if indices_path is None:
        records = iter_manifest_range(manifest, offsets, start, end)
    else:
        render_indices = np.load(indices_path, mmap_mode="r", allow_pickle=False)
        if render_indices.ndim != 1 or render_indices.dtype != np.dtype(np.uint64):
            raise RuntimeError(f"Invalid render index: {indices_path}")
        if not 0 <= start < end <= len(render_indices):
            raise ValueError(f"Invalid cache worker task interval: [{start},{end})")
        records = iter_manifest_indices(manifest, offsets, render_indices[start:end])
    for task_index, (index, record) in enumerate(records, start=1):
        t0 = time.time()
        value = renderer.render(record["location"], record["map_pose"])
        packed[index] = np.packbits(
            value.astype(np.uint8, copy=False).reshape(-1), bitorder="little"
        )
        dt = time.time() - t0
        if dt > 1.0:
            print(
                f"slow_sample index={index} location={record['location']} "
                f"pose={record['map_pose']} dt={dt:.2f}s",
                flush=True,
            )
        processed = task_index
        if processed % 50 == 0:
            elapsed = time.time() - t_start
            print(
                f"progress {processed}/{end - start} index={index} "
                f"elapsed={elapsed:.1f}s avg={elapsed/processed:.2f}s/sample",
                flush=True,
            )
    packed.flush()
    print(f"worker_done {start}:{end} total={time.time()-t_start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build row-aligned bit-packed MagicDrive static-map mmap cache"
    )
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--segment-size", type=int, default=500)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of isolated cache segments to render concurrently",
    )
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=300.0,
        help="Maximum seconds allowed for one isolated cache segment",
    )
    parser.add_argument(
        "--reuse-data-config",
        help=(
            "Copy exact clip_id/location/map_pose matches from this completed "
            "config's cache before rendering missing rows"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--end", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--indices-path", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.start is None or args.end is None:
            raise ValueError("cache worker requires --start and --end")
        _worker(
            args.data_config,
            args.split,
            args.start,
            args.end,
            args.indices_path,
        )
        return
    if (
        args.segment_size < 1
        or args.workers < 1
        or args.max_failures < 1
        or args.worker_timeout <= 0
    ):
        raise ValueError("segment-size/workers/max-failures/worker-timeout must be positive")

    config, map_config, manifest, output_dir, output, offsets_path = _context(
        args.data_config, args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    metadata_path = output_dir / f"{args.split}.json"
    progress_path = output_dir / f"{args.split}.progress.json"
    render_indices_path = output_dir / f"{args.split}.render_indices.npy"
    manifest_sha, record_count = build_manifest_index(manifest, offsets_path)
    if output.is_file() and metadata_path.is_file() and not args.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "manifest_sha256": manifest_sha,
            "records": record_count,
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
        print(f"reused_static_maps={record_count} output={output}", flush=True)
        return
    if output.is_file() or metadata_path.is_file():
        if not args.overwrite:
            raise RuntimeError(
                "Static-map cache is incomplete (data/metadata pair is missing); "
                f"use --overwrite: {output}"
            )
        output.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    progress: dict = {}
    if progress_path.is_file() and temp.is_file() and not args.overwrite:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("Static-map cache progress belongs to another manifest")
        if int(progress.get("plan_version", 1)) >= 2:
            if not render_indices_path.is_file():
                raise RuntimeError(
                    f"Static-map cache render plan is missing: {render_indices_path}"
                )
            next_task = int(progress["next_task"])
            completed_before_plan = int(progress.get("completed_before_plan", 0))
            reused_records = int(progress.get("reused_records", 0))
            reuse_source = progress.get("reuse_source")
            reuse_source_manifest_sha = progress.get(
                "reuse_source_manifest_sha256"
            )
        else:
            # Upgrade progress written by the pre-reuse cache builder.  Its
            # next_index is a contiguous, already-rendered manifest prefix.
            completed_before_plan = int(progress["next_index"])
            if args.reuse_data_config:
                render_indices, reuse_info = reuse_static_map_rows(
                    source_data_config=args.reuse_data_config,
                    split=args.split,
                    target_config=config,
                    target_map_config=map_config,
                    target_manifest=manifest,
                    target_offsets=offsets_path,
                    target_temp=temp,
                    skip_before=completed_before_plan,
                )
            else:
                render_indices = np.arange(
                    completed_before_plan, record_count, dtype=np.uint64
                )
                reuse_info = {}
            _save_npy_atomic(render_indices_path, render_indices)
            next_task = 0
            reused_records = int(reuse_info.get("reused_records", 0))
            reuse_source = reuse_info.get("reuse_source")
            reuse_source_manifest_sha = reuse_info.get(
                "reuse_source_manifest_sha256"
            )
    else:
        progress_path.unlink(missing_ok=True)
        packed = np.lib.format.open_memmap(
            temp, mode="w+", dtype=np.uint8, shape=(record_count, 40000)
        )
        packed.flush()
        del packed
        completed_before_plan = 0
        if args.reuse_data_config:
            render_indices, reuse_info = reuse_static_map_rows(
                source_data_config=args.reuse_data_config,
                split=args.split,
                target_config=config,
                target_map_config=map_config,
                target_manifest=manifest,
                target_offsets=offsets_path,
                target_temp=temp,
            )
        else:
            render_indices = np.arange(record_count, dtype=np.uint64)
            reuse_info = {}
        _save_npy_atomic(render_indices_path, render_indices)
        next_task = 0
        reused_records = int(reuse_info.get("reused_records", 0))
        reuse_source = reuse_info.get("reuse_source")
        reuse_source_manifest_sha = reuse_info.get("reuse_source_manifest_sha256")

    render_indices = np.load(render_indices_path, mmap_mode="r", allow_pickle=False)
    if render_indices.ndim != 1 or render_indices.dtype != np.dtype(np.uint64):
        raise RuntimeError(f"Invalid static-map render plan: {render_indices_path}")
    render_count = len(render_indices)
    if not 0 <= next_task <= render_count:
        raise RuntimeError(
            f"Invalid static-map progress: next_task={next_task}, tasks={render_count}"
        )
    if completed_before_plan + reused_records + render_count != record_count:
        raise RuntimeError(
            "Static-map cache plan does not cover the manifest: "
            f"completed={completed_before_plan}, reused={reused_records}, "
            f"render={render_count}, records={record_count}"
        )

    def write_progress() -> None:
        state = {
            "plan_version": 2,
            "manifest_sha256": manifest_sha,
            "next_task": next_task,
            "completed_before_plan": completed_before_plan,
            "reused_records": reused_records,
            "render_records": render_count,
        }
        if reuse_source is not None:
            state["reuse_source"] = reuse_source
            state["reuse_source_manifest_sha256"] = reuse_source_manifest_sha
        progress_temp = progress_path.with_suffix(progress_path.suffix + ".tmp")
        progress_temp.write_text(
            json.dumps(state, sort_keys=True) + "\n", encoding="utf-8"
        )
        progress_temp.replace(progress_path)

    write_progress()
    if reused_records:
        print(
            f"reused_static_map_rows={reused_records} "
            f"render_required={render_count} source={reuse_source}",
            flush=True,
        )

    failures = 0
    def run_segment(segment: tuple[int, int]):
        start, end = segment
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
            str(start),
            "--end",
            str(end),
            "--indices-path",
            str(render_indices_path),
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
            return subprocess.CompletedProcess(command, returncode=124)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while next_task < render_count:
            segments: list[tuple[int, int]] = []
            start = next_task
            for _ in range(args.workers):
                if start >= render_count:
                    break
                end = min(start + args.segment_size, render_count)
                segments.append((start, end))
                start = end
            results = list(executor.map(run_segment, segments))
            failed = [
                (segment, result.returncode)
                for segment, result in zip(segments, results)
                if result.returncode != 0
            ]
            if not failed:
                next_task = segments[-1][1]
                failures = 0
                write_progress()
                prepared = completed_before_plan + reused_records + next_task
                print(
                    f"cached_static_maps={prepared}/{record_count} "
                    f"rendered={completed_before_plan + next_task} "
                    f"reused={reused_records}",
                    flush=True,
                )
                continue
            failures += 1
            if failures >= args.max_failures:
                raise RuntimeError(
                    f"Static-map batch failed {failures} times: {failed}"
                )
            print(
                f"isolated_cache_failures={failed} "
                f"attempt={failures}/{args.max_failures}",
                flush=True,
            )

    temp.replace(output)
    progress_path.unlink(missing_ok=True)
    renderer = _renderer(config, map_config)
    metadata = {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "records": record_count,
        "map_shape": [8, 200, 200],
        "packed_bytes_per_map": 40000,
        "bitorder": "little",
        "classes": list(renderer.classes),
        "xbound": list(renderer.xbound),
        "ybound": list(renderer.ybound),
        "data_root": str(config["data_root"]),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "rendered_records": completed_before_plan + render_count,
        "reused_records": reused_records,
    }
    if reuse_source is not None:
        metadata["reuse_source"] = reuse_source
        metadata["reuse_source_manifest_sha256"] = reuse_source_manifest_sha
    metadata_temp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata_temp.replace(metadata_path)
    render_indices_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
