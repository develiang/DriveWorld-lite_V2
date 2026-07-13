"""Real-data V2 forward/sample smoke; never performs backward or optimizer steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from driveworld.config import load_yaml
from driveworld.data import NuScenesFrontDataset
from driveworld.models.factory import build_diffusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config",
        default="configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/single_view_stdit_rf_v2_local.yaml",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sample-steps", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("V2 real-data smoke requires CUDA")

    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    dataset = NuScenesFrontDataset(
        Path(data_config["manifest_dir"]) / f"{args.split}.jsonl",
        data_config["data_root"],
        tuple(data_config["resolution"]),
    )
    item = dataset[args.index]
    device = torch.device("cuda")
    model = build_diffusion(model_config, int(data_config["history_frames"])).to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    past = item["past_rgb"][None].to(device)
    future = item["future_rgb"][None].to(device)
    ego = item["future_ego"][None].to(device)
    valid = item["future_ego_valid"][None].to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        losses = model.training_loss(past, future, ego, valid)
        generated = model.sample(
            past,
            ego,
            valid,
            num_steps=args.sample_steps,
            sampler="heun",
        )
    torch.cuda.synchronize(device)
    report = {
        "clip_id": item["clip_id"],
        "model": type(model.denoiser).__name__,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "condition_history_frames": model.condition_history_frames,
        "history_latent_frames": model.vae.latent_frame_count(model.condition_history_frames),
        "future_latent_frames": model.vae.latent_frame_count(int(data_config["future_frames"])),
        "loss": float(losses["loss"]),
        "per_future_latent_loss": losses["per_future_latent_loss"].float().cpu().tolist(),
        "generated_shape": list(generated.shape),
        "finite": bool(torch.isfinite(generated).all()),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        "optimizer_steps": 0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
