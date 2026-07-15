from __future__ import annotations

from pathlib import Path

import torch

from .magicdrive_single_view_stdit import MagicDriveSingleViewSTDiT
from .mdd_condition_adapter import MDDConditionAdapter
from .pretrained import _unwrap_state_dict


def _resolve_dtype(dtype):
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    names = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return names[str(dtype).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported checkpoint dtype: {dtype}") from exc


def load_mdd_singleview_base(
    checkpoint: str | Path,
    *,
    device="cpu",
    dtype=torch.float32,
    model_kwargs: dict | None = None,
):
    """Materialize only the 1.2B single-view base from the 2.0B FP32 Stage-3 EMA.

    The model is first constructed on the meta device. Each retained source tensor is
    converted directly to the requested device/dtype, so no second target model or
    converted checkpoint file is required.
    """
    checkpoint = Path(checkpoint)
    dtype = _resolve_dtype(dtype)
    device = torch.device(device)
    kwargs = dict(model_kwargs or {})
    with torch.device("meta"):
        model = MagicDriveSingleViewSTDiT(**kwargs)
    target = model.state_dict()
    source = _unwrap_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    )

    missing = []
    mismatched = {}
    matched_keys = 0
    matched_numel = 0
    estimated_parameter_bytes = 0
    for name, target_tensor in target.items():
        source_tensor = source.get(name)
        if source_tensor is None:
            missing.append(name)
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            mismatched[name] = {
                "source": list(source_tensor.shape),
                "target": list(target_tensor.shape),
            }
            continue
        target_dtype = dtype if dtype is not None and source_tensor.is_floating_point() else None
        value = source_tensor.to(device=device, dtype=target_dtype)
        parent_name, leaf_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        if leaf_name in parent._parameters:
            old = parent._parameters[leaf_name]
            parent._parameters[leaf_name] = torch.nn.Parameter(
                value, requires_grad=bool(old.requires_grad)
            )
        elif leaf_name in parent._buffers:
            parent._buffers[leaf_name] = value
        else:
            raise RuntimeError(f"Cannot materialize unknown state entry: {name}")
        matched_keys += 1
        matched_numel += value.numel()
        estimated_parameter_bytes += value.numel() * value.element_size()
    if missing or mismatched:
        raise RuntimeError(
            f"MDD single-view checkpoint mismatch: missing={len(missing)}, "
            f"shape_mismatch={len(mismatched)}"
        )

    if matched_keys != len(target):
        raise RuntimeError(
            f"MDD materialization incomplete: matched={matched_keys}, target={len(target)}"
        )
    # PositionEmbedding2D.inv_freq is deliberately non-persistent in MagicDrive,
    # so it is not present in the EMA and must be materialized after meta loading.
    half = model.hidden_size // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half))
    model.pos_embed.inv_freq = inv_freq.to(device=device, dtype=torch.float32)
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "matched_keys": matched_keys,
        "matched_numel": matched_numel,
        "source_keys": len(source),
        "source_numel": sum(tensor.numel() for tensor in source.values()),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch.") if dtype is not None else "source",
        "estimated_parameter_bytes": estimated_parameter_bytes,
    }
    return model, report


def load_mdd_condition_adapter(
    checkpoint: str | Path,
    *,
    device="cpu",
    dtype=torch.float32,
    adapter_kwargs: dict | None = None,
):
    """Load Stage-3 camera/frame weights and retain a new zero-init kinematics branch."""
    checkpoint = Path(checkpoint)
    dtype = _resolve_dtype(dtype)
    device = torch.device(device)
    adapter = MDDConditionAdapter(**dict(adapter_kwargs or {})).to(device=device, dtype=dtype)
    source = _unwrap_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    )
    target = adapter.state_dict()
    pretrained_names = [
        name
        for name in target
        if name.startswith(("camera_embedder.", "frame_embedder.", "bbox_embedder."))
    ]
    missing = [name for name in pretrained_names if name not in source]
    mismatched = {
        name: {"source": list(source[name].shape), "target": list(target[name].shape)}
        for name in pretrained_names
        if name in source and tuple(source[name].shape) != tuple(target[name].shape)
    }
    if missing or mismatched:
        raise RuntimeError(
            f"MDD condition checkpoint mismatch: missing={len(missing)}, "
            f"shape_mismatch={len(mismatched)}"
        )
    compatible = {
        name: source[name].to(device=device, dtype=dtype) for name in pretrained_names
    }
    result = adapter.load_state_dict(compatible, strict=False)
    expected_new = {
        name for name in target if name.startswith("kinematics_embedder.")
    }
    if set(result.missing_keys) != expected_new or result.unexpected_keys:
        raise RuntimeError("Unexpected missing/unexpected keys while loading MDD conditions")
    return adapter, {
        "checkpoint": str(checkpoint.resolve()),
        "matched_keys": len(compatible),
        "matched_numel": sum(value.numel() for value in compatible.values()),
        "new_keys": len(expected_new),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
    }
