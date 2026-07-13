from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def _config_value(config, dotted_key):
    value = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def validate_checkpoint_compatibility(saved, expected, keys):
    mismatches = {}
    for key in keys:
        saved_value = _config_value(saved, key)
        expected_value = _config_value(expected, key)
        if saved_value != expected_value:
            mismatches[key] = {"checkpoint": saved_value, "current": expected_value}
    if mismatches:
        raise RuntimeError(f"Incompatible resume configuration: {mismatches}")


def _to_cpu(value):
    """Detach checkpoint tensors from CUDA before serialization for long-run stability."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    ema,
    step: int,
    config: dict,
    exclude_prefixes: tuple[str, ...] = (),
    include_names: tuple[str, ...] | None = None,
    scaler=None,
) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    include = set(include_names) if include_names is not None else None
    model_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if (
            name in include
            if include is not None
            else not any(name.startswith(prefix) for prefix in exclude_prefixes)
        )
    }
    state = {
        "step": step,
        "model": model_state,
        "excluded_model_prefixes": list(exclude_prefixes),
        "included_model_names": sorted(include) if include is not None else None,
        "optimizer": _to_cpu(optimizer.state_dict()),
        "scheduler": _to_cpu(scheduler.state_dict()) if scheduler is not None else None,
        "scaler": _to_cpu(scaler.state_dict()) if scaler is not None else None,
        "ema": _to_cpu(ema.state_dict()) if ema is not None else None,
        "config": config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    temp.replace(path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    ema=None,
    scaler=None,
    restore_rng=True,
    expected_config=None,
    compatibility_keys=(),
):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if expected_config is not None and compatibility_keys:
        validate_checkpoint_compatibility(
            state.get("config", {}), expected_config, compatibility_keys
        )
    saved_include = state.get("included_model_names")
    if saved_include is not None:
        expected_include = set(getattr(model, "checkpoint_include_names", ()))
        saved_include = set(saved_include)
        state_names = set(state["model"])
        if state_names != saved_include:
            raise RuntimeError(
                "Checkpoint model delta is incomplete: "
                f"declared={len(saved_include)} tensors={len(state_names)}"
            )
        if expected_include and saved_include != expected_include:
            raise RuntimeError(
                "Checkpoint model delta contract mismatch: "
                f"checkpoint={len(saved_include)} current={len(expected_include)}"
            )
    result = model.load_state_dict(state["model"], strict=False)
    excluded = tuple(
        dict.fromkeys(
            tuple(state.get("excluded_model_prefixes", ()))
            + tuple(getattr(model, "checkpoint_exclude_prefixes", ()))
        )
    )
    unexpected_missing = [
        key for key in result.missing_keys if not any(key.startswith(prefix) for prefix in excluded)
    ]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={unexpected_missing}, unexpected={result.unexpected_keys}"
        )
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if ema is not None and state["ema"] is not None:
        ema.load_state_dict(state["ema"])
    if restore_rng:
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state
