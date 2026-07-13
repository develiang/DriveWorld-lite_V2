from __future__ import annotations

import numpy as np


def psnr(prediction, target, data_range: float = 2.0) -> float:
    if hasattr(prediction, "detach"):
        mse = (prediction - target).square().mean().item()
    else:
        mse = float(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(data_range**2 / mse))


def ssim(prediction, target, data_range: float = 2.0) -> float:
    """Global SSIM fallback; use skimage/windowed SSIM for publication numbers."""
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().float().cpu().numpy()
        target = target.detach().float().cpu().numpy()
    x, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covariance = np.mean((x - mu_x) * (y - mu_y))
    return float(((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)))

