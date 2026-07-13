from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from driveworld.config import load_yaml
from driveworld.data import NuScenesLatentDataset
from driveworld.models.factory import build_diffusion
from driveworld.training.checkpoint import load_checkpoint
from driveworld.training.ema import EMA
from driveworld.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", default="configs/data/nuscenes_front_4x8_6hz_128x224.yaml")
    parser.add_argument("--model-config", default="configs/model/latent_diffusion_local_16gb.yaml")
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=16)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/timestep_diagnostics.json"))
    args = parser.parse_args()
    data_config, model_config = load_yaml(args.data_config), load_yaml(args.model_config)
    manifest = Path(data_config["manifest_dir"]) / "train.jsonl"
    dataset = NuScenesLatentDataset(
        manifest, args.latent_cache / "train.jsonl", allow_incomplete=True
    )
    loader = DataLoader(Subset(dataset, range(min(args.max_clips, len(dataset)))), batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_diffusion(model_config, int(data_config["history_frames"]), load_vae=False).to(device)
    ema = EMA(model.denoiser)
    state = load_checkpoint(args.checkpoint, model, ema=ema, restore_rng=False)
    if not args.raw:
        ema.copy_to(model.denoiser)
    model.eval()
    timesteps = [0, 10, 50, 100, 250, 500, 750, 900, 999]
    results = {}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for timestep in timesteps:
            values = []
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            for batch in loader:
                past, future = batch["past_latent"].to(device), batch["future_latent"].to(device)
                noise = torch.randn_like(torch.cat([past, future], dim=1))
                result = model.training_loss_latents(
                    past,
                    future,
                    batch["future_ego"].to(device),
                    batch["future_ego_valid"].to(device),
                    timesteps=torch.full((len(past),), timestep, device=device, dtype=torch.long),
                    noise=noise,
                )
                values.append(float(result["loss"]))
            results[str(timestep)] = sum(values) / len(values)
    report = {
        "checkpoint": str(args.checkpoint),
        "step": int(state["step"]),
        "weights": "raw" if args.raw else "ema",
        "loss_by_timestep": results,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
