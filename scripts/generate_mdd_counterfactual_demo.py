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
from driveworld.utils import write_json


MODES = ("original", "straight", "left", "right", "stop")


def _image(frame) -> Image.Image:
    value = frame.detach().float().cpu().clamp(-1, 1).numpy().transpose(1, 2, 0)
    return Image.fromarray(np.round((value + 1) * 127.5).astype(np.uint8))


def _save_gif(video, output: Path, duration_ms: int):
    frames = [_image(frame) for frame in video[0]]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _trajectory_summary(ego: np.ndarray):
    speed = np.linalg.norm(ego[:, 3:5], axis=1)
    return {
        "final_xy_m": ego[-1, :2].tolist(),
        "displacement_m": float(np.linalg.norm(ego[-1, :2])),
        "speed_min_mps": float(speed.min()),
        "speed_mean_mps": float(speed.mean()),
        "speed_max_mps": float(speed.max()),
        "final_yaw_degrees": float(np.rad2deg(ego[-1, 2])),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a fixed-seed V2-MDDiT Ego counterfactual comparison GIF"
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/v2_mdd_stage3_singleview_lora_6hz.yaml",
    )
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--turn-yaw-degrees", type=float, default=25.0)
    parser.add_argument("--raw", action="store_true", help="Do not apply adapter EMA")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mdd_counterfactual_demo"),
    )
    args = parser.parse_args()

    import torch

    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    sampling = model_config.get("sampling", {})
    num_steps = args.num_steps or int(sampling.get("num_steps", 30))
    guidance = (
        args.guidance_scale
        if args.guidance_scale is not None
        else float(sampling.get("guidance_scale", 2.0))
    )
    manifest = args.manifest or Path(data_config["manifest_dir"]) / "val.jsonl"
    dataset = NuScenesFrontDataset(
        manifest,
        data_config["data_root"],
        tuple(data_config["resolution"]),
        normalize_ego=False,
        static_map=data_config.get("static_map"),
    )
    item = dataset[args.index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V2-MDDiT counterfactual inference requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_diffusion(
        model_config,
        int(data_config["history_frames"]),
        device=device,
    )
    state = load_checkpoint(args.adapter_checkpoint, model, restore_rng=False)
    weights = "adapter_raw"
    if not args.raw and state.get("ema") is not None:
        model.adapter_ema_target.load_state_dict(state["ema"]["shadow"])
        weights = "adapter_ema"
    model.eval()

    batch = {
        key: value[None].to(device)
        for key, value in item.items()
        if isinstance(value, torch.Tensor)
    }
    original_ego = item["future_ego_raw"].numpy()
    valid = batch["future_ego_valid"]
    videos = {}
    trajectories = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = round(1000 / float(data_config["fps"]))
    with torch.inference_mode():
        for mode in MODES:
            trajectory = edit_trajectory(
                original_ego,
                mode,
                fps=float(data_config["fps"]),
                turn_yaw_degrees=args.turn_yaw_degrees,
            )
            future_ego = torch.from_numpy(trajectory)[None].to(device)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            video = model.sample(
                batch["past_rgb"],
                future_ego,
                valid,
                past_ego_raw=batch["past_ego_raw"],
                past_ego_valid=batch["past_ego_valid"],
                camera_parameters=batch.get("camera_parameters"),
                camera_valid=batch.get("camera_valid"),
                static_maps=batch.get("static_maps"),
                num_steps=num_steps,
                guidance_scale=guidance,
                generator=generator,
            ).cpu()
            videos[mode] = video
            trajectories[mode] = trajectory
            _save_gif(video, args.output_dir / f"{mode}.gif", duration_ms)

    labels = ("Anchor", "Ground truth", *MODES)
    anchor = _image(item["past_rgb"][-1])
    frames = []
    for frame_index, target in enumerate(item["future_rgb"]):
        images = [anchor, _image(target)]
        images.extend(_image(videos[mode][0, frame_index]) for mode in MODES)
        width, height = images[0].size
        canvas = Image.new("RGB", (width * len(images), height + 28), "black")
        draw = ImageDraw.Draw(canvas)
        for column, (label, image) in enumerate(zip(labels, images)):
            canvas.paste(image, (column * width, 28))
            draw.text((column * width + 5, 7), label, fill="white")
        frames.append(canvas)
    grid = args.output_dir / "counterfactual_grid.gif"
    frames[0].save(
        grid,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )

    reference_video = videos["original"].float()
    reference_ego = trajectories["original"]
    condition_differences = {}
    for mode in MODES[1:]:
        video_distance = float((videos[mode].float() - reference_video).abs().mean())
        trajectory_distance = float(np.abs(trajectories[mode] - reference_ego).mean())
        condition_differences[mode] = {
            "video_mean_abs": video_distance,
            "trajectory_mean_abs": trajectory_distance,
            "sensitivity": video_distance / max(trajectory_distance, 1e-8),
        }
    quality = horizon_report(reference_video, item["future_rgb"][None])
    report = {
        "clip_id": item["clip_id"],
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "checkpoint_step": int(state["step"]),
        "weights": weights,
        "seed": args.seed,
        "num_steps": num_steps,
        "guidance_scale": guidance,
        "turn_yaw_degrees": args.turn_yaw_degrees,
        "grid": str(grid),
        "source_trajectory_summary": _trajectory_summary(original_ego),
        "condition_differences": condition_differences,
        "trajectory_summaries": {
            mode: _trajectory_summary(value) for mode, value in trajectories.items()
        },
        "trajectories": {mode: value.tolist() for mode, value in trajectories.items()},
        "original_quality_by_frame": {
            key: value.detach().float().cpu().tolist() if value.ndim else float(value)
            for key, value in quality.items()
        },
        "cuda_peak_allocated": torch.cuda.max_memory_allocated(device),
    }
    write_json(args.output_dir / "metadata.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "trajectories"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
