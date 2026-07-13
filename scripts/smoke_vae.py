"""Load the local CogVideoX VAE and run a tiny deterministic reconstruction test."""

from __future__ import annotations

import argparse
import json
import time

import torch

from driveworld.config import load_yaml
from driveworld.models.video_vae import CogVideoXVAEAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/latent_diffusion_ego.yaml")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    # Some CPU-only PyTorch builds crash inside oneDNN's large Conv3D kernels.
    # This diagnostic favors correctness; production VAE work should use CUDA.
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False
    config = load_yaml(args.config)["vae"]
    start = time.perf_counter()
    vae = CogVideoXVAEAdapter(
        config["pretrained"],
        config.get("subfolder"),
        local_files_only=bool(config.get("local_files_only", True)),
    )
    load_seconds = time.perf_counter() - start
    value = torch.zeros(1, args.frames, 3, args.height, args.width)
    start = time.perf_counter()
    latent = vae.encode(value)
    reconstruction = vae.decode(latent, output_frames=args.frames)
    run_seconds = time.perf_counter() - start
    report = {
        "input_shape": list(value.shape),
        "latent_shape": list(latent.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "finite": bool(torch.isfinite(reconstruction).all()),
        "latent_channels": vae.latent_channels,
        "temporal_compression_ratio": vae.temporal_compression_ratio,
        "load_seconds": load_seconds,
        "encode_decode_seconds": run_seconds,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
