from __future__ import annotations

import argparse
import json

from driveworld.config import load_yaml
from driveworld.data import ClipConfig, build_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CAM_FRONT 8→16 clip manifests")
    parser.add_argument("--config", default="configs/data/nuscenes_front_8x16_6hz.yaml")
    args = parser.parse_args()
    raw_config = load_yaml(args.config)
    report = build_manifests(ClipConfig.from_dict(raw_config), raw_config)
    print(json.dumps(report["split_counts"], indent=2))
    print(f"total_clips={report['total_clips']}")


if __name__ == "__main__":
    main()

