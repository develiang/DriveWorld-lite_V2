from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None


def _require_video(value):
    if value.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {tuple(value.shape)}")


def per_frame_psnr(prediction, target, data_range: float = 2.0):
    _require_video(prediction)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    mse = (prediction.float() - target.float()).square().mean(dim=(0, 2, 3, 4))
    peak = torch.tensor(data_range**2, device=mse.device, dtype=mse.dtype)
    return 10 * torch.log10(peak / mse.clamp_min(1e-12))


def per_frame_mae(prediction, target):
    _require_video(prediction)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    return (prediction.float() - target.float()).abs().mean(dim=(0, 2, 3, 4))


def per_frame_edge_energy(video):
    _require_video(video)
    value = video.float()
    vertical = (value[..., 1:, :] - value[..., :-1, :]).abs().mean(dim=(0, 2, 3, 4))
    horizontal = (value[..., :, 1:] - value[..., :, :-1]).abs().mean(dim=(0, 2, 3, 4))
    return 0.5 * (vertical + horizontal)


def horizon_report(prediction, target):
    prediction_edges = per_frame_edge_energy(prediction)
    target_edges = per_frame_edge_energy(target)
    retention = prediction_edges / target_edges.clamp_min(1e-8)
    return {
        "psnr": per_frame_psnr(prediction, target),
        "mae": per_frame_mae(prediction, target),
        "prediction_edge": prediction_edges,
        "target_edge": target_edges,
        "edge_retention": retention,
        "last_to_first_prediction_edge": prediction_edges[-1]
        / prediction_edges[0].clamp_min(1e-8),
    }
