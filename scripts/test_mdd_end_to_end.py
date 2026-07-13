from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from driveworld.data import NuScenesFrontDataset
from driveworld.diffusion import MagicRectifiedFlowScheduler
from driveworld.models.magic_cogvideox_adapter import MagicCogVideoXVAEAdapter
from driveworld.models.mdd_checkpoint import (
    load_mdd_condition_adapter,
    load_mdd_singleview_base,
)
from driveworld.models.mdd_world_model import MDDI2VWorldModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data VAE -> MDDiT -> RF no-training smoke")
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained/MDDiT/ema.pt"))
    parser.add_argument("--vae", default="pretrained/vae")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/manifests/nuscenes-trainval-partial-front-8x16-6hz/val.jsonl"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/nuscenes-trainval"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--resolution", type=int, nargs=2, default=(64, 112))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control-depth", type=int, default=0)
    parser.add_argument("--static-map", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    denoiser, model_report = load_mdd_singleview_base(
        args.checkpoint,
        device=device,
        dtype="bf16",
        model_kwargs={"control_depth": args.control_depth},
    )
    condition, condition_report = load_mdd_condition_adapter(
        args.checkpoint, device=device, dtype="bf16"
    )
    vae = MagicCogVideoXVAEAdapter(
        args.vae,
        subfolder=None,
        local_files_only=True,
        posterior="sample",
    ).to(device=device, dtype=torch.bfloat16)
    scheduler = MagicRectifiedFlowScheduler(
        sample_method="logit_normal",
        use_timestep_transform=True,
        cog_style_transform=True,
    )
    world = MDDI2VWorldModel(vae, denoiser, condition, scheduler, fps=6.0)
    world.freeze_for_kinematics_adapter_training().eval()

    dataset = NuScenesFrontDataset(
        args.manifest,
        args.data_root,
        resolution=tuple(args.resolution),
        normalize_ego=False,
        static_map={"enabled": args.static_map},
    )
    sample = dataset[args.index]
    batch = {
        key: value[None].to(device)
        for key, value in sample.items()
        if isinstance(value, torch.Tensor)
    }
    metadata = {
        "height": torch.tensor([args.resolution[0]], device=device),
        "width": torch.tensor([args.resolution[1]], device=device),
        "num_frames": torch.tensor([17], device=device),
    }
    timestep = scheduler.transform_timesteps(torch.tensor([500.0], device=device), metadata)
    torch.manual_seed(42)
    with torch.inference_mode():
        result = world(
            past_rgb=batch["past_rgb"],
            future_rgb=batch["future_rgb"],
            past_ego_raw=batch["past_ego_raw"],
            future_ego_raw=batch["future_ego_raw"],
            past_ego_valid=batch["past_ego_valid"],
            future_ego_valid=batch["future_ego_valid"],
            camera_parameters=batch.get("camera_parameters"),
            camera_valid=batch.get("camera_valid"),
            static_maps=batch.get("static_maps"),
            timesteps=timestep,
        )
    report = {
        "clip_id": sample["clip_id"],
        "resolution": list(args.resolution),
        "model": model_report,
        "condition": condition_report,
        "latent_shape": list(result["latent_shape"]),
        "condition_shape": list(result["condition_shape"]),
        "camera_condition": (
            "calibrated_3x7" if "camera_parameters" in batch else "learned_unconditional"
        ),
        "control_depth": args.control_depth,
        "static_map": args.static_map,
        "prediction_shape": list(result["prediction"].shape),
        "loss": result["loss"].item(),
        "finite": bool(torch.isfinite(result["prediction"]).all()),
        "timestep": result["timesteps"].item(),
        "cuda_peak_allocated": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
