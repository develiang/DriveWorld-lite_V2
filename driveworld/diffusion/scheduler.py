from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = object


class LinearNoiseScheduler(nn.Module if torch is not None else object):
    def __init__(self, num_train_timesteps: int = 1000, beta_start=0.0001, beta_end=0.02):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        beta = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0).float()
        self.register_buffer("alpha_cumprod", alpha_cumprod, persistent=True)
        self.num_train_timesteps = num_train_timesteps

    def _coefficients(self, timesteps, ndim: int):
        alpha = self.alpha_cumprod[timesteps]
        shape = (len(timesteps),) + (1,) * (ndim - 1)
        return alpha.sqrt().view(shape), (1 - alpha).sqrt().view(shape)

    def add_noise(self, clean, noise, timesteps):
        sqrt_alpha, sqrt_one_minus = self._coefficients(timesteps, clean.ndim)
        return sqrt_alpha * clean + sqrt_one_minus * noise

    def velocity(self, clean, noise, timesteps):
        sqrt_alpha, sqrt_one_minus = self._coefficients(timesteps, clean.ndim)
        return sqrt_alpha * noise - sqrt_one_minus * clean

    def clean_from_velocity(self, noisy, velocity, timesteps):
        sqrt_alpha, sqrt_one_minus = self._coefficients(timesteps, noisy.ndim)
        return sqrt_alpha * noisy - sqrt_one_minus * velocity

    def noise_from_velocity(self, noisy, velocity, timesteps):
        sqrt_alpha, sqrt_one_minus = self._coefficients(timesteps, noisy.ndim)
        return sqrt_alpha * velocity + sqrt_one_minus * noisy

