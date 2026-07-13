from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from driveworld.config import load_yaml
from driveworld.data import NuScenesLatentDataset
from driveworld.models.factory import build_diffusion
from driveworld.training.checkpoint import load_checkpoint
from driveworld.training.ema import EMA
from driveworld.utils import write_json


def evaluate(model, loader, device, seed: int, repeats: int) -> float:
    values = []
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for repeat in range(repeats):
            torch.manual_seed(seed + repeat)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + repeat)
            for batch in loader:
                result = model(
                    past_latent=batch["past_latent"].to(device),
                    future_latent=batch["future_latent"].to(device),
                    future_ego=batch["future_ego"].to(device),
                    future_ego_valid=batch["future_ego_valid"].to(device),
                )
                values.append(float(result["loss"]))
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints with fixed latent/noise inputs")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--data-config", default="configs/data/nuscenes_front_4x8_6hz_128x224.yaml")
    parser.add_argument("--model-config", default="configs/model/latent_diffusion_local_16gb.yaml")
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--raw", action="store_true", help="Evaluate raw rather than EMA weights")
    parser.add_argument("--output", type=Path, default=Path("artifacts/diffusion_loss_report.json"))
    args = parser.parse_args()

    data_config, model_config = load_yaml(args.data_config), load_yaml(args.model_config)
    manifest_dir = Path(data_config["manifest_dir"])
    datasets = {
        split: NuScenesLatentDataset(
            manifest_dir / f"{split}.jsonl",
            args.latent_cache / f"{split}.jsonl",
            allow_incomplete=True,
        )
        for split in ["train", "val"]
    }
    loaders = {
        split: DataLoader(
            torch.utils.data.Subset(dataset, range(min(args.max_clips, len(dataset)))),
            batch_size=1,
            shuffle=False,
        )
        for split, dataset in datasets.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_diffusion(
        model_config, int(data_config["history_frames"]), load_vae=False
    ).to(device)
    rows = []
    for checkpoint in args.checkpoints:
        ema = EMA(model.denoiser)
        state = load_checkpoint(checkpoint, model, ema=ema, restore_rng=False)
        if not args.raw:
            ema.copy_to(model.denoiser)
        row = {"checkpoint": str(checkpoint), "step": int(state["step"])}
        for split, loader in loaders.items():
            row[f"{split}_loss"] = evaluate(model, loader, device, args.seed, args.repeats)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "weights": "raw" if args.raw else "ema",
        "clips_per_split": args.max_clips,
        "repeats": args.repeats,
        "seed": args.seed,
        "results": rows,
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()

