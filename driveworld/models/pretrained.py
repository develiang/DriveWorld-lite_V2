from __future__ import annotations

from pathlib import Path


def _unwrap_state_dict(value):
    if not isinstance(value, dict):
        raise TypeError("Checkpoint must contain a state dict")
    for key in ("state_dict", "model", "module"):
        nested = value.get(key)
        if isinstance(nested, dict) and nested:
            value = nested
            break
    return value


def _candidate_names(name: str):
    yield name
    current = name
    for prefix in ("module.", "model.", "denoiser."):
        if current.startswith(prefix):
            current = current[len(prefix) :]
            yield current
    if name.startswith("module.denoiser."):
        yield name[len("module.denoiser.") :]


def audit_pretrained_state(model, checkpoint_state: dict):
    source = _unwrap_state_dict(checkpoint_state)
    target = model.state_dict()
    compatible = {}
    shape_mismatch = {}
    unused = []
    for source_name, value in source.items():
        target_name = next((name for name in _candidate_names(source_name) if name in target), None)
        if target_name is None:
            unused.append(source_name)
            continue
        if tuple(value.shape) != tuple(target[target_name].shape):
            shape_mismatch[source_name] = {
                "source": list(value.shape),
                "target": list(target[target_name].shape),
            }
            continue
        compatible[target_name] = value
    target_parameters = sum(value.numel() for value in target.values())
    matched_parameters = sum(target[name].numel() for name in compatible)
    missing = sorted(set(target) - set(compatible))
    return {
        "compatible": compatible,
        "matched_keys": len(compatible),
        "target_keys": len(target),
        "matched_parameters": matched_parameters,
        "target_parameters": target_parameters,
        "parameter_coverage": matched_parameters / max(target_parameters, 1),
        "missing": missing,
        "unused": unused,
        "shape_mismatch": shape_mismatch,
    }


def load_pretrained_denoiser(model, path: str | Path, min_coverage: float = 0.0):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    report = audit_pretrained_state(model, state)
    if report["parameter_coverage"] < min_coverage:
        raise RuntimeError(
            f"Pretrained coverage {report['parameter_coverage']:.3f} is below required {min_coverage:.3f}"
        )
    model.load_state_dict(report["compatible"], strict=False)
    return {key: value for key, value in report.items() if key != "compatible"}
