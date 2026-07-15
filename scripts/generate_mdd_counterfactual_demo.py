from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from driveworld.config import load_yaml
from driveworld.control import edit_trajectory
from driveworld.data import NuScenesFrontDataset
from driveworld.evaluation import control_gate_report, horizon_report, motion_report, pair_report
from driveworld.models.factory import build_diffusion
from driveworld.training.checkpoint import load_checkpoint
from driveworld.utils import write_json


DEFAULT_MODES = (
    "original",
    "straight",
    "left",
    "right",
    "stop",
    "hold",
    "shuffle",
    "invalid",
    "zero_kinematics",
)
EDITED_MODES = {"original", "straight", "left", "right", "stop", "hold", "zero"}
ALL_MODES = (*DEFAULT_MODES, "zero")


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


def _save_grid(item, videos, modes, output: Path, duration_ms: int):
    labels = ("Anchor", "Ground truth", *modes)
    anchor = _image(item["past_rgb"][-1])
    frames = []
    for frame_index, target in enumerate(item["future_rgb"]):
        images = [anchor, _image(target)]
        images.extend(_image(videos[mode][0, frame_index]) for mode in modes)
        width, height = images[0].size
        canvas = Image.new("RGB", (width * len(images), height + 28), "black")
        draw = ImageDraw.Draw(canvas)
        for column, (label, image) in enumerate(zip(labels, images)):
            canvas.paste(image, (column * width, 28))
            draw.text((column * width + 5, 7), label, fill="white")
        frames.append(canvas)
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


def _json_tensor(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
        return value.tolist() if value.ndim else float(value)
    return value


def _condition_variant(item, shuffle_item, mode: str, fps: float, turn_yaw_degrees: float):
    original = item["future_ego_raw"].numpy().copy()
    valid = item["future_ego_valid"].numpy().copy()
    source = {"kind": "edited", "clip_id": item["clip_id"]}
    if mode in EDITED_MODES:
        trajectory = edit_trajectory(
            original,
            mode,
            fps=fps,
            turn_yaw_degrees=turn_yaw_degrees,
        )
        if mode in {"hold", "zero"}:
            valid = np.ones_like(valid, dtype=np.bool_)
            source["kind"] = "zero_ego_hold"
        return trajectory, valid, source
    if mode == "shuffle":
        if shuffle_item is None:
            raise ValueError("shuffle mode requires a second validation sample")
        return (
            shuffle_item["future_ego_raw"].numpy().copy(),
            shuffle_item["future_ego_valid"].numpy().copy(),
            {"kind": "shuffled", "clip_id": shuffle_item["clip_id"]},
        )
    if mode == "invalid":
        return original, np.zeros_like(valid, dtype=np.bool_), {
            "kind": "future_ego_invalid",
            "clip_id": item["clip_id"],
        }
    if mode == "zero_kinematics":
        trajectory = original.copy()
        trajectory[:, 3:] = 0
        return trajectory, valid, {
            "kind": "zero_kinematics_pose_preserved",
            "clip_id": item["clip_id"],
        }
    raise ValueError(f"Unknown counterfactual mode: {mode}")


def _evaluate_case(
    *,
    model,
    item,
    shuffle_item,
    modes,
    seed: int,
    num_steps: int,
    guidance: float,
    turn_yaw_degrees: float,
    fps: float,
    output_dir: Path,
    weights: str,
    checkpoint,
    checkpoint_step,
    motion_backend: str,
    gate_thresholds,
    save_gifs: bool,
):
    import torch

    device = next(model.parameters()).device
    batch = {
        key: value[None].to(device)
        for key, value in item.items()
        if isinstance(value, torch.Tensor)
    }
    videos = {}
    trajectories = {}
    trajectory_valid = {}
    condition_sources = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = round(1000 / fps)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for mode in modes:
            trajectory, valid, source = _condition_variant(
                item,
                shuffle_item,
                mode,
                fps,
                turn_yaw_degrees,
            )
            future_ego = torch.from_numpy(trajectory)[None].to(device)
            future_valid = torch.from_numpy(valid)[None].to(device)
            generator = torch.Generator(device=device).manual_seed(seed)
            video = model.sample(
                batch["past_rgb"],
                future_ego,
                future_valid,
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
            trajectory_valid[mode] = valid
            condition_sources[mode] = source
            if save_gifs:
                _save_gif(video, output_dir / f"{mode}.gif", duration_ms)

    grid = output_dir / "counterfactual_grid.gif"
    if save_gifs:
        _save_grid(item, videos, modes, grid, duration_ms)

    anchor = item["past_rgb"][-1]
    motion = {
        mode: motion_report(video, anchor=anchor, backend=motion_backend)
        for mode, video in videos.items()
    }
    reference_video = videos["original"].float()
    reference_ego = trajectories["original"]
    pairs = {}
    condition_differences = {}
    for mode in modes:
        if mode == "original":
            continue
        pair = pair_report(videos[mode], reference_video)
        pairs[mode] = pair
        trajectory_distance = float(np.abs(trajectories[mode] - reference_ego).mean())
        condition_differences[mode] = {
            **pair,
            "trajectory_mean_abs": trajectory_distance,
            "sensitivity": float(pair["video_mae"]) / max(trajectory_distance, 1e-8),
        }
    pairwise = {}
    hold_mode = "hold" if "hold" in videos else "zero"
    for first, second in (
        ("stop", "straight"),
        (hold_mode, "straight"),
        ("left", "right"),
    ):
        if first in videos and second in videos:
            pairwise[f"{first}_vs_{second}"] = pair_report(videos[first], videos[second])

    finite = {mode: bool(torch.isfinite(video).all()) for mode, video in videos.items()}
    gate = control_gate_report(
        motion,
        pairs,
        finite=finite,
        thresholds=gate_thresholds,
    )
    quality = horizon_report(reference_video, item["future_rgb"][None])
    report = {
        "format": "mdd-control-eval-v1",
        "clip_id": item["clip_id"],
        "shuffle_clip_id": shuffle_item["clip_id"] if shuffle_item is not None else None,
        "adapter_checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_step": checkpoint_step,
        "weights": weights,
        "seed": seed,
        "num_steps": num_steps,
        "guidance_scale": guidance,
        "turn_yaw_degrees": turn_yaw_degrees,
        "modes": list(modes),
        "grid": str(grid) if save_gifs else None,
        "motion_backend": next(iter(motion.values()))["backend"],
        "finite": finite,
        "gate": gate,
        "condition_sources": condition_sources,
        "condition_differences": condition_differences,
        "pairwise_differences": pairwise,
        "motion": motion,
        "trajectory_summaries": {
            mode: _trajectory_summary(value) for mode, value in trajectories.items()
        },
        "original_quality_by_frame": {
            key: _json_tensor(value) for key, value in quality.items()
        },
        "trajectories": {mode: value.tolist() for mode, value in trajectories.items()},
        "trajectory_valid": {
            mode: value.astype(bool).tolist() for mode, value in trajectory_valid.items()
        },
        "cuda_peak_allocated": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    write_json(output_dir / "metadata.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-noise V2-MDDiT counterfactuals and evaluate control gates"
        )
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/v2_mdd_stage3_singleview_lora_6hz.yaml",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        help="Omit to evaluate the pretrained step-zero/zero-init adapter baseline",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=0, help="Legacy single-index option")
    parser.add_argument("--indices", type=int, nargs="+", help="Evaluate multiple val indices")
    parser.add_argument("--shuffle-index", type=int)
    parser.add_argument("--seed", type=int, default=42, help="Legacy single-seed option")
    parser.add_argument("--seeds", type=int, nargs="+", help="Evaluate multiple fixed seeds")
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--turn-yaw-degrees", type=float, default=25.0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ALL_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument(
        "--motion-backend",
        choices=["auto", "farneback", "frame_mae"],
        default="auto",
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=Path("configs/eval/mdd_control_gate_pilot.yaml"),
    )
    parser.add_argument("--raw", action="store_true", help="Do not apply adapter EMA")
    parser.add_argument("--no-gifs", action="store_true", help="Write JSON metrics only")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mdd_control_eval"),
    )
    args = parser.parse_args()
    if "original" not in args.modes:
        raise SystemExit("--modes must include original as the fixed-noise reference")
    if args.raw and args.adapter_checkpoint is None:
        raise SystemExit("--raw requires --adapter-checkpoint")

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
    indices = args.indices or [args.index]
    seeds = args.seeds or [args.seed]
    for index in indices:
        if not 0 <= index < len(dataset):
            raise SystemExit(f"Validation index out of range: {index} (size={len(dataset)})")
    gate_thresholds = load_yaml(args.gate_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V2-MDDiT control evaluation requires CUDA")
    torch.cuda.empty_cache()
    model = build_diffusion(
        model_config,
        int(data_config["history_frames"]),
        device=device,
    )
    checkpoint_step = None
    weights = "pretrained_zero_init_adapter"
    if args.adapter_checkpoint is not None:
        state = load_checkpoint(args.adapter_checkpoint, model, restore_rng=False)
        checkpoint_step = int(state["step"])
        weights = "adapter_raw"
        if not args.raw and state.get("ema") is not None:
            model.adapter_ema_target.load_state_dict(state["ema"]["shadow"])
            weights = "adapter_ema"
    model.eval()

    case_count = len(indices) * len(seeds)
    summaries = []
    for index in indices:
        item = dataset[index]
        shuffle_index = args.shuffle_index
        if shuffle_index is None:
            shuffle_index = (index + 1) % len(dataset)
        if not 0 <= shuffle_index < len(dataset):
            raise SystemExit(
                f"Shuffle index out of range: {shuffle_index} (size={len(dataset)})"
            )
        shuffle_item = dataset[shuffle_index] if "shuffle" in args.modes else None
        for seed in seeds:
            case_dir = args.output_dir
            if case_count > 1:
                case_dir = args.output_dir / f"clip-{index:06d}" / f"seed-{seed:010d}"
            report = _evaluate_case(
                model=model,
                item=item,
                shuffle_item=shuffle_item,
                modes=args.modes,
                seed=seed,
                num_steps=num_steps,
                guidance=guidance,
                turn_yaw_degrees=args.turn_yaw_degrees,
                fps=float(data_config["fps"]),
                output_dir=case_dir,
                weights=weights,
                checkpoint=args.adapter_checkpoint,
                checkpoint_step=checkpoint_step,
                motion_backend=args.motion_backend,
                gate_thresholds=gate_thresholds,
                save_gifs=not args.no_gifs,
            )
            summaries.append(
                {
                    "clip_id": report["clip_id"],
                    "index": index,
                    "seed": seed,
                    "metadata": str(case_dir / "metadata.json"),
                    "grid": report["grid"],
                    "motion_backend": report["motion_backend"],
                    "gate_status": report["gate"]["status"],
                    "failed_checks": report["gate"]["failed_checks"],
                    "unavailable_checks": report["gate"]["unavailable_checks"],
                }
            )

    counts = {
        status: sum(case["gate_status"] == status for case in summaries)
        for status in ("pass", "fail", "incomplete")
    }
    selected_backends = sorted({case["motion_backend"] for case in summaries})
    summary = {
        "format": "mdd-control-eval-summary-v1",
        "adapter_checkpoint": str(args.adapter_checkpoint) if args.adapter_checkpoint else None,
        "checkpoint_step": checkpoint_step,
        "weights": weights,
        "motion_backend": (
            selected_backends[0] if len(selected_backends) == 1 else selected_backends
        ),
        "gate_config": str(args.gate_config),
        "cases": summaries,
        "status_counts": counts,
        "pass_rate": counts["pass"] / max(len(summaries), 1),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
