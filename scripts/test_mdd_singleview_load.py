from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from driveworld.models.mdd_checkpoint import load_mdd_singleview_base


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-only smoke test for MDD single-view base")
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained/MDDiT/ema.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--forward-smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model, report = load_mdd_singleview_base(
        args.checkpoint, device=device, dtype=args.dtype
    )
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    if any(value.device.type == "meta" for value in parameters + buffers):
        raise RuntimeError("Model still contains meta tensors after checkpoint loading")
    report.update(
        {
            "parameter_device": str(parameters[0].device),
            "parameter_dtype": str(parameters[0].dtype).removeprefix("torch."),
            "cuda_allocated": torch.cuda.memory_allocated(device) if device.type == "cuda" else None,
            "cuda_reserved": torch.cuda.memory_reserved(device) if device.type == "cuda" else None,
            "cuda_peak_allocated": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
    )
    if args.forward_smoke:
        model.eval()
        dtype = parameters[0].dtype
        latent = torch.randn(1, 16, 5, 8, 8, device=device, dtype=dtype)
        condition = torch.zeros(1, 5, 1, 1152, device=device, dtype=dtype)
        x_mask = torch.tensor([[False, True, True, True, True]], device=device)
        with torch.inference_mode():
            output = model(
                latent,
                torch.tensor([500.0], device=device),
                condition,
                fps=12.0,
                height=64,
                width=64,
                x_mask=x_mask,
            )
        report["forward"] = {
            "input_shape": list(latent.shape),
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype).removeprefix("torch."),
            "finite": bool(torch.isfinite(output).all()),
            "mean": output.mean().item(),
            "std": output.std().item(),
        }
        report["cuda_peak_allocated"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
