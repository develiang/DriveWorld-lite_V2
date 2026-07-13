from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from driveworld.config import load_yaml
from driveworld.data import NuScenesFrontDataset
from driveworld.models.factory import build_diffusion
from driveworld.training.checkpoint import load_checkpoint


def _image(frame):
    value = frame.detach().float().cpu().clamp(-1, 1).numpy().transpose(1, 2, 0)
    return Image.fromarray(np.round((value + 1) * 127.5).astype(np.uint8))


def _save_gif(video, output, duration_ms):
    frames = [_image(frame) for frame in video[0]]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="V2-MDDiT single-image 30-step Euler/CFG inference"
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml",
    )
    parser.add_argument(
        "--model-config", default="configs/model/v2_mdd_stage3_singleview.yaml"
    )
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        help="Optional H W override for a low-resolution inference smoke",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--raw", action="store_true", help="Do not apply adapter EMA")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mdd_inference"))
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
    resolution = tuple(args.resolution or data_config["resolution"])
    dataset = NuScenesFrontDataset(
        manifest,
        data_config["data_root"],
        resolution,
        normalize_ego=False,
        static_map=data_config.get("static_map"),
    )
    item = dataset[args.index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The Stage-3 MDDiT inference entry currently requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_diffusion(
        model_config,
        int(data_config["history_frames"]),
        device=device,
    )
    checkpoint_step = None
    weights = "pretrained_zero_init_adapter"
    if args.adapter_checkpoint:
        state = load_checkpoint(args.adapter_checkpoint, model, restore_rng=False)
        checkpoint_step = int(state["step"])
        if not args.raw and state.get("ema") is not None:
            model.adapter_ema_target.load_state_dict(state["ema"]["shadow"])
            weights = "adapter_ema"
        else:
            weights = "adapter_raw"
    model.eval()
    batch = {
        key: value[None].to(device)
        for key, value in item.items()
        if isinstance(value, torch.Tensor)
    }
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        video = model.sample(
            batch["past_rgb"],
            batch["future_ego_raw"],
            batch["future_ego_valid"],
            past_ego_raw=batch["past_ego_raw"],
            past_ego_valid=batch["past_ego_valid"],
            camera_parameters=batch.get("camera_parameters"),
            camera_valid=batch.get("camera_valid"),
            static_maps=batch.get("static_maps"),
            num_steps=num_steps,
            guidance_scale=guidance,
            generator=generator,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gif = args.output_dir / f"clip-{args.index:04d}.gif"
    _save_gif(video, gif, round(1000 / float(data_config["fps"])))
    report = {
        "clip_id": item["clip_id"],
        "adapter_checkpoint": str(args.adapter_checkpoint) if args.adapter_checkpoint else None,
        "checkpoint_step": checkpoint_step,
        "weights": weights,
        "seed": args.seed,
        "num_steps": num_steps,
        "guidance_scale": guidance,
        "resolution": list(resolution),
        "video_shape": list(video.shape),
        "finite": bool(torch.isfinite(video).all()),
        "gif": str(gif),
        "cuda_peak_allocated": torch.cuda.max_memory_allocated(device),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
