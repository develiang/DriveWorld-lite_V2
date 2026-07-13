from __future__ import annotations

import argparse
import json
from pathlib import Path

from driveworld.control import edit_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible counterfactual Ego JSON")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/counterfactual_trajectories.json"))
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    record = records[args.index]
    trajectories = {
        mode: edit_trajectory(record["future_ego"], mode, args.fps).tolist()
        for mode in ["original", "straight", "left", "right", "stop"]
    }
    output = {"clip_id": record["clip_id"], "fps": args.fps, "trajectories": trajectories}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

