from __future__ import annotations

import argparse
import json

from driveworld.config import load_yaml
from driveworld.data import ClipConfig, build_manifests
from driveworld.data.can_interpolator import CanBusInterpolator
from driveworld.data.nuscenes_tables import NuScenesTables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CAM_FRONT 8→16 clip manifests")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/data/nuscenes_front_8x16_6hz.yaml"],
    )
    args = parser.parse_args()
    shared_key = None
    tables = None
    can = None
    for config_path in args.config:
        raw_config = load_yaml(config_path)
        config = ClipConfig.from_dict(raw_config)
        key = (config.data_root, config.version, config.camera)
        if key != shared_key:
            tables = NuScenesTables(
                config.data_root, config.version, camera_filter=config.camera
            )
            can = CanBusInterpolator(config.data_root / "can_bus")
            shared_key = key
        report = build_manifests(config, raw_config, tables=tables, can=can)
        print(f"config={config_path}")
        print(json.dumps(report["split_counts"], indent=2))
        print(f"total_clips={report['total_clips']}")


if __name__ == "__main__":
    main()
