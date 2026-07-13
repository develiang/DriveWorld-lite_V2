from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import torch


CROSS_VIEW_MARKERS = (
    ".cross_view_attn.",
    ".mva_proj.",
    ".norm3.",
    ".scale_shift_table_mva",
)


def unwrap_state_dict(value: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping, got {type(value).__name__}")
    for key in ("state_dict", "model", "module"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and nested:
            value = nested
            break
    non_tensors = [name for name, tensor in value.items() if not isinstance(tensor, torch.Tensor)]
    if non_tensors:
        preview = ", ".join(str(name) for name in non_tensors[:5])
        raise TypeError(f"State dict contains non-tensor values: {preview}")
    return value


def classify_key(name: str) -> str:
    if any(marker in name for marker in CROSS_VIEW_MARKERS):
        return "cross_view"
    if name.startswith("base_blocks_s."):
        return "base_spatial"
    if name.startswith("base_blocks_t."):
        return "base_temporal"
    if name.startswith("control_blocks_s."):
        return "control_spatial"
    if name.startswith("control_blocks_t."):
        return "control_temporal"
    if name.startswith("y_embedder."):
        return "text"
    if name.startswith(("camera_embedder.", "frame_embedder.", "bbox_embedder.")):
        return "condition_embedder"
    if name.startswith(("controlnet_cond_", "before_proj.")):
        return "map_control"
    if name.startswith("x_control_embedder."):
        return "control_input"
    if name.startswith("final_layer."):
        return "final_layer"
    if name.startswith(
        ("x_embedder.", "t_embedder.", "t_block.", "fps_embedder.", "base_token", "rope.")
    ):
        return "core_embedding"
    return "other"


def tensor_description(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "numel": tensor.numel(),
    }


def infer_architecture(state: Mapping[str, torch.Tensor]) -> dict[str, object]:
    architecture: dict[str, object] = {}
    patch_weight = state.get("x_embedder.proj.weight")
    if patch_weight is not None and patch_weight.ndim == 5:
        architecture.update(
            {
                "hidden_size": patch_weight.shape[0],
                "in_channels": patch_weight.shape[1],
                "patch_size": list(patch_weight.shape[2:]),
            }
        )
    block_indices: dict[str, set[int]] = defaultdict(set)
    for name in state:
        root = name.split(".", 1)[0]
        if root not in {"base_blocks_s", "base_blocks_t", "control_blocks_s", "control_blocks_t"}:
            continue
        parts = name.split(".", 2)
        if len(parts) > 1 and parts[1].isdigit():
            block_indices[root].add(int(parts[1]))
    architecture.update({f"{root}_depth": len(indices) for root, indices in block_indices.items()})
    qkv_weight = state.get("base_blocks_s.0.attn.qkv.weight")
    if qkv_weight is not None and qkv_weight.ndim == 2:
        architecture["qkv_shape"] = list(qkv_weight.shape)
    final_weight = state.get("final_layer.linear.weight")
    if final_weight is not None:
        architecture["final_linear_shape"] = list(final_weight.shape)
    return architecture


def summarize_state_dict(state: Mapping[str, torch.Tensor], sample_limit: int = 8) -> dict[str, object]:
    dtype_tensors: Counter[str] = Counter()
    dtype_numel: Counter[str] = Counter()
    group_tensors: Counter[str] = Counter()
    group_numel: Counter[str] = Counter()
    root_tensors: Counter[str] = Counter()
    root_numel: Counter[str] = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    total_numel = 0
    tensor_bytes = 0

    for name, tensor in state.items():
        dtype = str(tensor.dtype).removeprefix("torch.")
        group = classify_key(name)
        root = name.split(".", 1)[0]
        numel = tensor.numel()
        total_numel += numel
        tensor_bytes += numel * tensor.element_size()
        dtype_tensors[dtype] += 1
        dtype_numel[dtype] += numel
        group_tensors[group] += 1
        group_numel[group] += numel
        root_tensors[root] += 1
        root_numel[root] += numel
        if len(samples[group]) < sample_limit:
            samples[group].append(name)

    def rows(tensor_counts: Counter[str], numel_counts: Counter[str]) -> dict[str, dict[str, object]]:
        return {
            name: {
                "tensors": tensor_counts[name],
                "numel": numel_counts[name],
                "fraction": numel_counts[name] / max(total_numel, 1),
            }
            for name in sorted(tensor_counts)
        }

    critical_names = (
        "base_token",
        "x_embedder.proj.weight",
        "t_embedder.mlp.0.weight",
        "fps_embedder.mlp.0.weight",
        "base_blocks_s.0.attn.qkv.weight",
        "base_blocks_t.0.attn.qkv.weight",
        "final_layer.linear.weight",
    )
    return {
        "tensor_count": len(state),
        "total_numel": total_numel,
        "tensor_bytes": tensor_bytes,
        "architecture": infer_architecture(state),
        "dtypes": rows(dtype_tensors, dtype_numel),
        "groups": rows(group_tensors, group_numel),
        "roots": rows(root_tensors, root_numel),
        "group_key_samples": dict(sorted(samples.items())),
        "critical_tensors": {
            name: tensor_description(state[name]) for name in critical_names if name in state
        },
    }


def summarize_mapping(
    source: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    matched = []
    missing = []
    shape_mismatch = {}
    for name, target_tensor in target.items():
        source_tensor = source.get(name)
        if source_tensor is None:
            missing.append(name)
        elif tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatch[name] = {
                "source": list(source_tensor.shape),
                "target": list(target_tensor.shape),
            }
        else:
            matched.append(name)

    unused = sorted(set(source) - set(target))
    source_numel = sum(tensor.numel() for tensor in source.values())
    target_numel = sum(tensor.numel() for tensor in target.values())
    matched_numel = sum(target[name].numel() for name in matched)
    unused_groups: Counter[str] = Counter()
    unused_group_tensors: Counter[str] = Counter()
    for name in unused:
        group = classify_key(name)
        unused_groups[group] += source[name].numel()
        unused_group_tensors[group] += 1
    return {
        "matched_keys": len(matched),
        "target_keys": len(target),
        "matched_numel": matched_numel,
        "target_numel": target_numel,
        "source_numel": source_numel,
        "target_coverage": matched_numel / max(target_numel, 1),
        "source_coverage": matched_numel / max(source_numel, 1),
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "unused_keys": len(unused),
        "unused_groups": {
            name: {"tensors": unused_group_tensors[name], "numel": unused_groups[name]}
            for name in sorted(unused_groups)
        },
    }


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_checkpoint(
    path: Path, include_sha256: bool = False, target: str | None = None
) -> dict[str, object]:
    stat = path.stat()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = unwrap_state_dict(checkpoint)
    report: dict[str, object] = {
        "format": "mdd-stage3-ema-audit-v1",
        "checkpoint": str(path.resolve()),
        "file_bytes": stat.st_size,
        "sha256": sha256_file(path) if include_sha256 else None,
    }
    report.update(summarize_state_dict(state))
    if target == "singleview-base":
        from driveworld.models.magicdrive_single_view_stdit import MagicDriveSingleViewSTDiT

        with torch.device("meta"):
            model = MagicDriveSingleViewSTDiT()
        report["mapping"] = {
            "target": target,
            **summarize_mapping(state, model.state_dict()),
        }
    elif target is not None:
        raise ValueError(f"Unknown audit target: {target}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of a MagicDrive Stage-3 EMA")
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained/MDDiT/ema.pt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sha256", action="store_true", help="Hash the full checkpoint file")
    parser.add_argument("--target", choices=["singleview-base"])
    args = parser.parse_args()

    report = audit_checkpoint(
        args.checkpoint, include_sha256=args.sha256, target=args.target
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
