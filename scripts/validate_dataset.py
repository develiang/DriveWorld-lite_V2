from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from driveworld.utils import write_json


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_manifest(path: Path, data_root: Path, check_images: int = 100) -> dict:
    records = load_manifest(path)
    build_report_path = path.parent / "build_report.json"
    if build_report_path.exists():
        build_config = json.loads(build_report_path.read_text(encoding="utf-8"))["config"]
        expected_history = int(build_config["history_frames"])
        expected_future = int(build_config["future_frames"])
    elif records:
        expected_history = len(records[0].get("past_ego", []))
        expected_future = len(records[0].get("future_ego", []))
    else:
        expected_history = expected_future = 0
    expected_total = expected_history + expected_future
    errors: list[str] = []
    scenes: Counter[str] = Counter()
    time_errors: list[float] = []
    steering_valid: list[float] = []
    rng = np.random.default_rng(42)
    image_check_indices = set(
        rng.choice(len(records), min(check_images, len(records)), replace=False).tolist()
    ) if records else set()
    checked_images = 0

    for index, record in enumerate(records):
        prefix = f"line {index + 1} ({record.get('clip_id', '?')})"
        scenes[record.get("scene_name", "?")] += 1
        paths = record.get("image_paths", [])
        target_t = np.asarray(record.get("target_timestamps_us", []), dtype=np.int64)
        image_t = np.asarray(record.get("image_timestamps_us", []), dtype=np.int64)
        past_ego = np.asarray(record.get("past_ego", []))
        future_ego = np.asarray(record.get("future_ego", []))
        past_valid = np.asarray(record.get("past_ego_valid", []), dtype=bool)
        future_valid = np.asarray(record.get("future_ego_valid", []), dtype=bool)
        if len(paths) != expected_total or len(target_t) != expected_total or len(image_t) != expected_total:
            errors.append(f"{prefix}: expected {expected_total} frames/timestamps")
        if past_ego.shape != (expected_history, 9) or future_ego.shape != (expected_future, 9):
            errors.append(f"{prefix}: invalid ego shapes {past_ego.shape}/{future_ego.shape}")
        if past_valid.shape != (expected_history, 9) or future_valid.shape != (expected_future, 9):
            errors.append(f"{prefix}: invalid mask shapes")
        if len(target_t) and (np.any(np.diff(target_t) <= 0) or np.any(np.diff(image_t) <= 0)):
            errors.append(f"{prefix}: non-monotonic timestamps")
        if len(target_t) == len(image_t) and len(target_t):
            time_errors.extend((np.abs(target_t - image_t) / 1000.0).tolist())
        if past_valid.shape == (expected_history, 9) and future_valid.shape == (expected_future, 9):
            steering_valid.append(float(np.r_[past_valid[:, 8], future_valid[:, 8]].mean()))
        if not np.isfinite(past_ego).all() or not np.isfinite(future_ego).all():
            errors.append(f"{prefix}: non-finite ego values")
        if index in image_check_indices:
            for relative in paths:
                image_path = data_root / relative
                if not image_path.exists():
                    errors.append(f"{prefix}: missing {relative}")
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    checked_images += 1
                except Exception as exc:  # Pillow exposes several decoder exception types.
                    errors.append(f"{prefix}: unreadable {relative}: {exc}")

    return {
        "manifest": str(path),
        "clips": len(records),
        "scenes": dict(scenes),
        "checked_images": checked_images,
        "camera_error_ms": {
            "mean": float(np.mean(time_errors)) if time_errors else None,
            "p95": float(np.percentile(time_errors, 95)) if time_errors else None,
            "max": float(np.max(time_errors)) if time_errors else None,
        },
        "steering_valid_fraction": float(np.mean(steering_valid)) if steering_valid else None,
        "error_count": len(errors),
        "errors": errors[:200],
        "valid": not errors and bool(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated clip manifests")
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/nuscenes-mini"))
    parser.add_argument("--check-images", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_validation.json"))
    args = parser.parse_args()
    reports = [
        validate_manifest(path, args.data_root, check_images=args.check_images)
        for path in args.manifests
    ]
    scene_to_split: dict[str, str] = {}
    leakage: list[str] = []
    for report in reports:
        split = Path(report["manifest"]).stem
        for scene in report["scenes"]:
            if scene in scene_to_split and scene_to_split[scene] != split:
                leakage.append(scene)
            scene_to_split[scene] = split
    output = {"reports": reports, "scene_leakage": sorted(set(leakage))}
    output["valid"] = all(x["valid"] for x in reports) and not leakage
    write_json(args.output, output)
    print(json.dumps(output, indent=2))
    if not output["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
