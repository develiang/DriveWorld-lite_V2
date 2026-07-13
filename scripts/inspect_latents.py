from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from driveworld.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cached VAE latent distribution")
    parser.add_argument("cache_index", type=Path)
    parser.add_argument("--max-clips", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/latent_stats.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.cache_index.read_text(encoding="utf-8").splitlines() if line]
    moments = []
    for row in rows[: args.max_clips]:
        cached = torch.load(args.cache_index.parent / row["path"], map_location="cpu", weights_only=True)
        moments.extend([cached["past"].float(), cached["future"].float()])
    if not moments:
        raise RuntimeError("No cached tensors found")
    values = torch.cat([value.permute(1, 0, 2, 3).reshape(value.shape[1], -1) for value in moments], dim=1)
    report = {
        "clips": min(len(rows), args.max_clips),
        "finite": bool(torch.isfinite(values).all()),
        "global": {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        },
        "channel_mean": values.mean(dim=1).tolist(),
        "channel_std": values.std(dim=1).tolist(),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

