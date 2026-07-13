"""Repeatedly encode real clips to exercise the online VAE without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from driveworld.config import load_yaml
from driveworld.data import NuScenesFrontDataset
from driveworld.models.video_vae import CogVideoXVAEAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config",
        default="configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/latent_diffusion_local_quality.yaml",
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This stress test requires CUDA")
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    dataset = NuScenesFrontDataset(
        Path(data_config["manifest_dir"]) / f"{args.split}.jsonl",
        data_config["data_root"],
        tuple(data_config["resolution"]),
    )
    vae_config = model_config["vae"]
    device = torch.device("cuda")
    vae = CogVideoXVAEAdapter(
        vae_config["pretrained"],
        vae_config.get("subfolder"),
        local_files_only=bool(vae_config.get("local_files_only", True)),
    ).to(device)
    torch.cuda.reset_peak_memory_stats(device)

    for iteration in range(args.iterations):
        item = dataset[iteration % len(dataset)]
        past = item["past_rgb"][None].to(device, non_blocking=True)
        future = item["future_rgb"][None].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            past_latent = vae.encode(past)
            future_latent = vae.encode(future)
        if not torch.isfinite(past_latent).all() or not torch.isfinite(future_latent).all():
            raise FloatingPointError(f"Non-finite VAE output at iteration {iteration}")
        del past, future, past_latent, future_latent
        if (iteration + 1) % args.log_every == 0 or iteration == 0:
            torch.cuda.synchronize(device)
            print(
                json.dumps(
                    {
                        "completed": iteration + 1,
                        "iterations": args.iterations,
                        "allocated_gb": round(torch.cuda.memory_allocated(device) / 2**30, 3),
                        "reserved_gb": round(torch.cuda.memory_reserved(device) / 2**30, 3),
                        "peak_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                    }
                ),
                flush=True,
            )

    torch.cuda.empty_cache()
    print("online VAE stress test passed", flush=True)


if __name__ == "__main__":
    main()
