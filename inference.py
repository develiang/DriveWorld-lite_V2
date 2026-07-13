from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from driveworld.config import load_yaml
from driveworld.control import edit_trajectory
from driveworld.data import NuScenesFrontDataset
from driveworld.models.factory import build_baseline, build_diffusion
from driveworld.training.checkpoint import load_checkpoint
from driveworld.training.ema import EMA


def save_video_gif(video, output: Path) -> None:
    value = video.detach().float().cpu().numpy()[0]
    frames = [Image.fromarray(np.clip((frame.transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)) for frame in value]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=167, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual Ego-conditioned inference")
    parser.add_argument("--task", choices=["baseline", "diffusion"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", default="configs/data/nuscenes_front_8x16_6hz.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--trajectory", choices=["original", "straight", "left", "right", "stop"], default="original")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--sampler", help="DDPM: ddim/legacy; Rectified Flow: euler/heun")
    parser.add_argument("--output", type=Path, default=Path("artifacts/inference.gif"))
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA")
    args = parser.parse_args()

    import torch

    data_config, model_config = load_yaml(args.data_config), load_yaml(args.model_config)
    manifest = args.manifest or Path(data_config["manifest_dir"]) / "val.jsonl"
    dataset = NuScenesFrontDataset(manifest, data_config["data_root"], tuple(data_config["resolution"]))
    item = dataset[args.index]
    future = edit_trajectory(item["future_ego_raw"].numpy(), args.trajectory, float(data_config["fps"]))
    item["future_ego"] = torch.from_numpy(
        dataset.normalize_ego(future, item["future_ego_valid"].numpy())
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (build_baseline(model_config) if args.task == "baseline" else build_diffusion(model_config, int(data_config["history_frames"]))).to(device)
    ema_target = model if args.task == "baseline" else model.denoiser
    ema = EMA(ema_target)
    load_checkpoint(args.checkpoint, model, ema=ema, restore_rng=False)
    if not args.no_ema:
        ema.copy_to(ema_target)
    model.eval()
    torch.manual_seed(args.seed)
    with torch.no_grad():
        inputs = {k: v[None].to(device) for k, v in item.items() if hasattr(v, "to")}
        if args.task == "baseline":
            video = model(inputs["past_rgb"], inputs["future_ego"], inputs["future_ego_valid"])
        else:
            sample_kwargs = {"num_steps": args.num_steps, "guidance": args.guidance}
            if args.sampler:
                sample_kwargs["sampler"] = args.sampler
            video = model.sample(
                inputs["past_rgb"], inputs["future_ego"], inputs["future_ego_valid"], **sample_kwargs
            )
    save_video_gif(video, args.output)
    metadata = {
        "clip_id": item["clip_id"],
        "trajectory": args.trajectory,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "guidance": args.guidance,
        "sampler": args.sampler or getattr(model, "default_sampler", "default"),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
