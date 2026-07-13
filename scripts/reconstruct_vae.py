"""Reconstruct real nuScenes frames with the frozen local VAE; no training is performed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from driveworld.config import load_yaml
from driveworld.data import NuScenesFrontDataset
from driveworld.models.video_vae import CogVideoXVAEAdapter
from driveworld.utils import write_json


def _to_image(value) -> Image.Image:
    array = value.detach().float().cpu().numpy().transpose(1, 2, 0)
    array = np.clip((array + 1) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/nuscenes_front_8x16_6hz.yaml")
    parser.add_argument("--model-config", default="configs/model/latent_diffusion_ego.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/vae_reconstruction.png"))
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False
    data_config, model_config = load_yaml(args.data_config), load_yaml(args.model_config)
    manifest = args.manifest or Path(data_config["manifest_dir"]) / "val.jsonl"
    dataset = NuScenesFrontDataset(
        manifest,
        data_config["data_root"],
        tuple(data_config["resolution"]),
    )
    item = dataset[args.index]
    full_clip = torch.cat([item["past_rgb"], item["future_rgb"]], dim=0)
    if args.frames < 1 or args.frames > len(full_clip):
        raise ValueError(f"--frames must be within [1,{len(full_clip)}]")
    original = full_clip[: args.frames][None]
    vae_config = model_config["vae"]
    vae = CogVideoXVAEAdapter(
        vae_config["pretrained"],
        vae_config.get("subfolder"),
        local_files_only=bool(vae_config.get("local_files_only", True)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    original = original.to(device)
    latent = vae.encode(original)
    reconstruction = vae.decode(latent, output_frames=args.frames)
    mse = (original - reconstruction).square().mean().item()
    psnr = float(10 * np.log10(4.0 / mse)) if mse else float("inf")

    panels = []
    for frame in range(args.frames):
        source, restored = _to_image(original[0, frame]), _to_image(reconstruction[0, frame])
        canvas = Image.new("RGB", (source.width * 2, source.height + 28), "black")
        canvas.paste(source, (0, 28))
        canvas.paste(restored, (source.width, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), "Original", fill="white")
        draw.text((source.width + 8, 7), f"CogVideoX VAE reconstruction | PSNR {psnr:.2f} dB", fill="white")
        panels.append(canvas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if len(panels) == 1:
        panels[0].save(args.output)
    else:
        gif = args.output.with_suffix(".gif")
        panels[0].save(gif, save_all=True, append_images=panels[1:], duration=167, loop=0)
        args.output = gif
    report = {
        "clip_id": item["clip_id"],
        "input_shape": list(original.shape),
        "latent_shape": list(latent.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "mse": mse,
        "psnr": psnr,
        "output": str(args.output),
        "device": str(device),
    }
    write_json(args.output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
