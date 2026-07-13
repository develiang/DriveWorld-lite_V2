from __future__ import annotations

import argparse
from pathlib import Path

from driveworld.visualization import render_manifest_clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/nuscenes-mini"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/clip.gif"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    print(render_manifest_clip(args.manifest, args.data_root, args.output, args.index))


if __name__ == "__main__":
    main()

