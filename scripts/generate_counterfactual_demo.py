from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from driveworld.config import load_yaml
from driveworld.control import edit_trajectory
from driveworld.data import NuScenesFrontDataset
from driveworld.evaluation import horizon_report
from driveworld.models.factory import build_diffusion
from driveworld.training.checkpoint import load_checkpoint
from driveworld.training.ema import EMA
from driveworld.utils import write_json


MODES = ["original", "straight", "left", "right", "stop"]


def tensor_image(value) -> Image.Image:
    array = value.detach().float().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(np.clip((array + 1) * 127.5, 0, 255).astype(np.uint8))


def save_gif(video, output: Path):
    frames = [tensor_image(frame) for frame in video[0]]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=167, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-seed Ego counterfactual grid")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", default="configs/data/nuscenes_front_4x8_6hz_128x224.yaml")
    parser.add_argument("--model-config", default="configs/model/latent_diffusion_local_16gb.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument(
        "--sampler",
        help="Sampler override (DDPM: ddim/legacy; Rectified Flow: euler/heun)",
    )
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--turn-yaw-degrees", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/counterfactual_demo"))
    args = parser.parse_args()

    import torch

    data_config, model_config = load_yaml(args.data_config), load_yaml(args.model_config)
    manifest = args.manifest or Path(data_config["manifest_dir"]) / "train.jsonl"
    dataset = NuScenesFrontDataset(
        manifest, data_config["data_root"], tuple(data_config["resolution"])
    )
    item = dataset[args.index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_diffusion(model_config, int(data_config["history_frames"])).to(device)
    ema = EMA(model.denoiser)
    state = load_checkpoint(args.checkpoint, model, ema=ema, restore_rng=False)
    if not args.raw:
        ema.copy_to(model.denoiser)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    past = item["past_rgb"][None].to(device)
    valid = item["future_ego_valid"].numpy()
    videos = {}
    trajectories = {}
    with torch.no_grad():
        for mode in MODES:
            raw = edit_trajectory(item["future_ego_raw"].numpy(), mode, float(data_config["fps"]), turn_yaw_degrees=args.turn_yaw_degrees)
            normalized = dataset.normalize_ego(raw, valid)
            ego = torch.from_numpy(normalized)[None].to(device)
            ego_valid = item["future_ego_valid"][None].to(device)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            sample_kwargs = {"num_steps": args.num_steps, "guidance": args.guidance}
            if args.sampler:
                sample_kwargs["sampler"] = args.sampler
            videos[mode] = model.sample(past, ego, ego_valid, **sample_kwargs).cpu()
            trajectories[mode] = raw.tolist()
            save_gif(videos[mode], args.output_dir / f"{mode}.gif")

    labels = ["Last history", "Ground truth"] + MODES
    history_image = tensor_image(item["past_rgb"][-1])
    ground_truth = item["future_rgb"]
    frames = []
    for frame_index in range(len(ground_truth)):
        images = [history_image, tensor_image(ground_truth[frame_index])]
        images.extend(tensor_image(videos[mode][0, frame_index]) for mode in MODES)
        width, height = images[0].size
        canvas = Image.new("RGB", (width * len(images), height + 28), "black")
        draw = ImageDraw.Draw(canvas)
        for column, (label, image) in enumerate(zip(labels, images)):
            canvas.paste(image, (column * width, 28))
            draw.text((column * width + 5, 7), label, fill="white")
        frames.append(canvas)
    grid = args.output_dir / "counterfactual_grid.gif"
    frames[0].save(grid, save_all=True, append_images=frames[1:], duration=167, loop=0)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(state["step"]),
        "weights": "raw" if args.raw else "ema",
        "clip_id": item["clip_id"],
        "seed": args.seed,
        "num_steps": args.num_steps,
        "guidance": args.guidance,
        "sampler": args.sampler or getattr(model, "default_sampler", "default"),
        "grid": str(grid),
        "condition_differences": {},
        "trajectories": trajectories,
    }
    reference_video = videos["original"].float()
    quality = horizon_report(reference_video, ground_truth[None])
    report["original_quality_by_frame"] = {
        key: value.detach().float().cpu().tolist() if value.ndim else float(value)
        for key, value in quality.items()
    }
    reference_trajectory = np.asarray(trajectories["original"], dtype=np.float32)
    for mode in MODES[1:]:
        video_distance = float((videos[mode].float() - reference_video).abs().mean())
        trajectory_distance = float(
            np.abs(np.asarray(trajectories[mode], dtype=np.float32) - reference_trajectory).mean()
        )
        report["condition_differences"][mode] = {
            "video_mean_abs": video_distance,
            "trajectory_mean_abs": trajectory_distance,
            "sensitivity": video_distance / max(trajectory_distance, 1e-8),
        }
    write_json(args.output_dir / "metadata.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "trajectories"}, indent=2))


if __name__ == "__main__":
    main()
