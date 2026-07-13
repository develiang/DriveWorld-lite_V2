from __future__ import annotations

import argparse
import json
from pathlib import Path

from driveworld.data.nuscenes_tables import NuScenesTables


def enrich_manifest(path: Path, tables: NuScenesTables, camera: str) -> dict:
    temp = path.with_suffix(path.suffix + ".camera.tmp")
    parameters_by_scene = {}
    count = 0
    with path.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            scene = record["scene_name"]
            if scene not in parameters_by_scene:
                camera_record = tables.camera_records(scene, camera)[0]
                parameters_by_scene[scene] = tables.magicdrive_camera_parameter(camera_record).tolist()
            record["schema_version"] = max(int(record.get("schema_version", 1)), 2)
            record["camera_parameter"] = parameters_by_scene[scene]
            record["camera_parameter_valid"] = True
            target.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    temp.replace(path)
    return {"manifest": str(path), "records": count, "scenes": len(parameters_by_scene)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Add MagicDrive 3x7 camera parameters to JSONL")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("manifests", type=Path, nargs="+")
    args = parser.parse_args()
    tables = NuScenesTables(args.data_root, args.version, camera_filter=args.camera)
    reports = [enrich_manifest(path, tables, args.camera) for path in args.manifests]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
