from __future__ import annotations

import argparse
import json
from pathlib import Path

from driveworld.data.nuscenes_tables import NuScenesTables


def enrich_manifest(path: Path, tables: NuScenesTables, camera: str) -> dict:
    records_by_scene = {}
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with path.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            scene_name = record["scene_name"]
            if scene_name not in records_by_scene:
                records_by_scene[scene_name] = {
                    item["filename"]: item
                    for item in tables.camera_records(scene_name, camera)
                }
            history = len(record["past_ego"])
            anchor_path = record["image_paths"][history - 1]
            try:
                camera_record = records_by_scene[scene_name][anchor_path]
            except KeyError as exc:
                raise KeyError(
                    f"{path}:{line_number}: anchor image is absent from {camera} chain: "
                    f"{anchor_path}"
                ) from exc
            record["schema_version"] = max(int(record.get("schema_version", 0)), 3)
            record["location"] = tables.scene_location(scene_name)
            record["map_pose"] = tables.magicdrive_map_pose(camera_record).tolist()
            target.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    temp.replace(path)
    return {"manifest": str(path), "records": count, "scenes": len(records_by_scene)}


def main():
    parser = argparse.ArgumentParser(
        description="Add MagicDrive static-map location/lidar pose to existing manifests"
    )
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--camera", default="CAM_FRONT")
    args = parser.parse_args()
    tables = NuScenesTables(args.data_root, args.version, camera_filter=args.camera)
    reports = [enrich_manifest(path, tables, args.camera) for path in args.manifests]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
