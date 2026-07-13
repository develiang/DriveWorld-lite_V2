from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from driveworld.models.mdd_checkpoint import (
    load_mdd_condition_adapter,
    load_mdd_singleview_base,
)


def read_jsonl_record(path: Path, index: int) -> dict:
    if index < 0:
        raise ValueError("index must be non-negative")
    with path.open(encoding="utf-8") as stream:
        for current, line in enumerate(stream):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Manifest index {index} is out of range: {path}")


def record_ego(record: dict, device, dtype):
    anchor = record["past_ego"][-1:]
    future = record["future_ego"]
    anchor_valid = record["past_ego_valid"][-1:]
    future_valid = record["future_ego_valid"]
    ego = torch.tensor([anchor + future], device=device, dtype=dtype)
    valid = torch.tensor([anchor_valid + future_valid], device=device, dtype=torch.bool)
    return ego, valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditioned Stage-3 single-view forward smoke")
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained/MDDiT/ema.pt"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/manifests/nuscenes-trainval-partial-front-8x16-6hz/val.jsonl"
        ),
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--control-depth", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model, model_report = load_mdd_singleview_base(
        args.checkpoint,
        device=device,
        dtype=args.dtype,
        model_kwargs={"control_depth": args.control_depth},
    )
    adapter, condition_report = load_mdd_condition_adapter(
        args.checkpoint, device=device, dtype=args.dtype
    )
    model.eval()
    adapter.eval()
    dtype = model.x_embedder.proj.weight.dtype
    record = read_jsonl_record(args.manifest, args.index)
    ego, ego_valid = record_ego(record, device, dtype)
    zero_ego = ego.clone()
    zero_ego[..., :3] = 0
    latent = torch.randn(1, 16, 5, 8, 8, device=device, dtype=dtype)
    x_mask = torch.tensor([[False, True, True, True, True]], device=device)
    timestep = torch.tensor([500.0], device=device)
    with torch.inference_mode():
        condition = adapter(ego, ego_valid, base_token=model.base_token)
        zero_condition = adapter(zero_ego, ego_valid, base_token=model.base_token)
        output = model(
            latent,
            timestep,
            condition,
            fps=6.0,
            height=64,
            width=64,
            x_mask=x_mask,
        )
        zero_output = model(
            latent,
            timestep,
            zero_condition,
            fps=6.0,
            height=64,
            width=64,
            x_mask=x_mask,
        )
    report = {
        "checkpoint": model_report,
        "condition_checkpoint": condition_report,
        "clip_id": record["clip_id"],
        "camera_condition": "learned_unconditional_camera",
        "control_depth": args.control_depth,
        "ego_shape": list(ego.shape),
        "condition_shape": list(condition.shape),
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()),
        "pose_condition_mean_abs_delta": (output - zero_output).abs().mean().item(),
        "cuda_peak_allocated": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
