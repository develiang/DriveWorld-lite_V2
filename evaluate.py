from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from driveworld.config import load_yaml
from driveworld.data import NuScenesFrontDataset
from driveworld.evaluation import frame_difference_error, psnr, ssim
from driveworld.models.factory import build_baseline
from driveworld.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-protocol validation")
    parser.add_argument("--task", choices=["last-frame", "baseline"], default="last-frame")
    parser.add_argument("--data-config", default="configs/data/nuscenes_front_8x16_6hz.yaml")
    parser.add_argument("--model-config", default="configs/model/unet3d_baseline.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-clips", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    data_config = load_yaml(args.data_config)
    manifest = Path(data_config["manifest_dir"]) / "val.jsonl"
    dataset = NuScenesFrontDataset(
        manifest,
        data_config["data_root"],
        tuple(data_config["resolution"]),
        return_numpy=args.task == "last-frame",
    )
    metrics = {"psnr": [], "ssim": [], "temporal": []}
    model = device = None
    if args.task == "baseline":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for baseline evaluation")
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_baseline(load_yaml(args.model_config)).to(device).eval()
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])

    for index in range(min(args.max_clips, len(dataset))):
        item = dataset[index]
        target = item["future_rgb"]
        if args.task == "last-frame":
            prediction = np.repeat(item["past_rgb"][-1:], len(target), axis=0)
            prediction, target = prediction[None], target[None]
        else:
            import torch

            with torch.no_grad():
                prediction = model(
                    item["past_rgb"][None].to(device),
                    item["future_ego"][None].to(device),
                    item["future_ego_valid"][None].to(device),
                )
            target = target[None].to(device)
        metrics["psnr"].append(psnr(prediction, target))
        metrics["ssim"].append(ssim(prediction, target))
        metrics["temporal"].append(frame_difference_error(prediction, target))

    report = {
        "task": args.task,
        "clips": len(metrics["psnr"]),
        "metrics": {name: float(np.mean(values)) for name, values in metrics.items()},
        "note": "SSIM is the dependency-free global fallback, not publication windowed SSIM.",
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

