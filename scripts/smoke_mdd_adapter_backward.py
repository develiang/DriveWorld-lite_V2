from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

from driveworld.config import load_yaml
from driveworld.data import NuScenesFrontDataset
from driveworld.models.factory import build_diffusion


def main():
    parser = argparse.ArgumentParser(
        description="One V2-MDDiT adapter backward without optimizer construction or step"
    )
    parser.add_argument(
        "--data-config",
        default="configs/data/nuscenes_front_8x16_6hz_trainval_partial.yaml",
    )
    parser.add_argument(
        "--model-config", default="configs/model/v2_mdd_stage3_singleview.yaml"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--resolution", type=int, nargs=2, default=(64, 112))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-details", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full Stage-3 backward smoke")
    device = torch.device("cuda")
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    manifest = args.manifest or Path(data_config["manifest_dir"]) / "train.jsonl"
    dataset = NuScenesFrontDataset(
        manifest,
        data_config["data_root"],
        tuple(args.resolution),
        normalize_ego=False,
        static_map=data_config.get("static_map"),
    )
    item = dataset[args.index]
    batch = {
        key: value[None].to(device)
        for key, value in item.items()
        if isinstance(value, torch.Tensor)
    }
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_diffusion(
        model_config,
        int(data_config["history_frames"]),
        device=device,
    ).train()
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(
            past_rgb=batch["past_rgb"],
            future_rgb=batch["future_rgb"],
            past_ego_raw=batch["past_ego_raw"],
            future_ego_raw=batch["future_ego_raw"],
            past_ego_valid=batch["past_ego_valid"],
            future_ego_valid=batch["future_ego_valid"],
            camera_parameters=batch.get("camera_parameters"),
            camera_valid=batch.get("camera_valid"),
        )
    result["loss"].backward()
    gradients = {
        name: {
            "finite": bool(parameter.grad is not None and torch.isfinite(parameter.grad).all()),
            "norm": float(parameter.grad.float().norm()) if parameter.grad is not None else None,
        }
        for name, parameter in trainable.items()
    }
    gradient_groups = {}
    for name, value in gradients.items():
        if name.startswith("condition_adapter.kinematics_embedder."):
            group = "action_adapter"
        elif name.endswith("lora_A"):
            group = "lora_A"
        elif name.endswith("lora_B"):
            group = "lora_B"
        elif name.endswith("scale_shift_table"):
            group = "adaln"
        else:
            group = "other"
        summary = gradient_groups.setdefault(
            group, {"parameters": 0, "finite": 0, "nonzero": 0, "norm_sum": 0.0}
        )
        summary["parameters"] += 1
        summary["finite"] += int(value["finite"])
        summary["nonzero"] += int(value["norm"] is not None and value["norm"] > 0)
        summary["norm_sum"] += float(value["norm"] or 0.0)
    report = {
        "clip_id": item["clip_id"],
        "resolution": list(args.resolution),
        "optimizer_constructed": False,
        "optimizer_step": False,
        "loss": float(result["loss"].detach()),
        "loss_finite": bool(torch.isfinite(result["loss"])),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "gradient_groups": gradient_groups,
        "all_gradients_finite": all(value["finite"] for value in gradients.values()),
        "cuda_peak_allocated": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved": torch.cuda.max_memory_reserved(device),
        "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if args.gradient_details:
        report["gradients"] = gradients
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
