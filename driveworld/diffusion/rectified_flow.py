from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = object


class RectifiedFlowScheduler(nn.Module if torch is not None else object):
    """Continuous flow with t=0 as noise and t=1 as clean data."""

    def __init__(self, sample_method: str = "logit_normal", logit_mean: float = 0.0, logit_std: float = 1.0):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        if sample_method not in {"uniform", "logit_normal"}:
            raise ValueError(f"Unknown Rectified Flow timestep method: {sample_method}")
        self.sample_method = sample_method
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)

    def sample_timesteps(self, batch_size: int, device):
        if self.sample_method == "uniform":
            return torch.rand(batch_size, device=device)
        logits = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
        return logits.sigmoid()

    @staticmethod
    def _coefficient(timesteps, ndim: int):
        return timesteps.view((len(timesteps),) + (1,) * (ndim - 1))

    def interpolate(self, clean, noise, timesteps):
        time = self._coefficient(timesteps, clean.ndim)
        return (1 - time) * noise + time * clean

    @staticmethod
    def velocity(clean, noise):
        return clean - noise

    def clean_from_velocity(self, value, velocity, timesteps):
        time = self._coefficient(timesteps, value.ndim)
        return value + (1 - time) * velocity

    def noise_from_velocity(self, value, velocity, timesteps):
        time = self._coefficient(timesteps, value.ndim)
        return value - time * velocity


class MaskedVideoRectifiedFlow(nn.Module if torch is not None else object):
    """Anchor/history-conditioned Rectified Flow in frozen VAE latent space."""

    def __init__(
        self,
        vae,
        denoiser,
        scheduler: RectifiedFlowScheduler,
        history_frames: int = 8,
        condition_history_frames: int | None = None,
        condition_dropout: float = 0.1,
        default_sampler: str = "heun",
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.vae = vae
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.history_frames = int(history_frames)
        self.condition_history_frames = int(condition_history_frames or history_frames)
        if not 1 <= self.condition_history_frames <= self.history_frames:
            raise ValueError("condition_history_frames must be within [1, history_frames]")
        if default_sampler not in {"euler", "heun"}:
            raise ValueError("default_sampler must be euler or heun")
        self.condition_dropout = float(condition_dropout)
        self.default_sampler = default_sampler

    def _select_history_rgb(self, past_rgb):
        if past_rgb.shape[1] < self.condition_history_frames:
            raise ValueError(
                f"Need {self.condition_history_frames} history RGB frames, got {past_rgb.shape[1]}"
            )
        return past_rgb[:, -self.condition_history_frames :]

    def _masks(self, latent, history_latent_frames: int):
        batch, frames, _, height, width = latent.shape
        history = latent.new_zeros(batch, frames, 1, height, width)
        history[:, :history_latent_frames] = 1
        return history, 1 - history

    @staticmethod
    def _model_timesteps(timesteps):
        # Reuse standard diffusion timestep embeddings at a useful numerical scale.
        return timesteps * 1000.0

    def training_loss(self, past_rgb, future_rgb, future_ego, future_ego_valid):
        selected_past = self._select_history_rgb(past_rgb)
        with torch.inference_mode():
            past_clean = self.vae.encode(selected_past)
            future_clean = self.vae.encode(future_rgb)
        return self.training_loss_latents(
            past_clean.clone(), future_clean.clone(), future_ego, future_ego_valid
        )

    def training_loss_latents(
        self,
        past_clean,
        future_clean,
        future_ego,
        future_ego_valid,
        timesteps=None,
        noise=None,
    ):
        expected_history = self.vae.latent_frame_count(self.condition_history_frames)
        if past_clean.shape[1] != expected_history:
            raise ValueError(
                f"V2 expected {expected_history} history latent frames for "
                f"condition_history_frames={self.condition_history_frames}, got {past_clean.shape[1]}. "
                "Rebuild the latent cache with the V2 model config."
            )
        clean = torch.cat([past_clean, future_clean], dim=1)
        history_frames = past_clean.shape[1]
        history_mask, future_mask = self._masks(clean, history_frames)
        if timesteps is None:
            timesteps = self.scheduler.sample_timesteps(len(clean), clean.device)
        else:
            timesteps = timesteps.to(device=clean.device, dtype=clean.dtype)
        if noise is None:
            noise = torch.randn_like(clean)
        interpolated = self.scheduler.interpolate(clean, noise, timesteps)
        mixed = history_mask * clean + future_mask * interpolated
        known = history_mask * clean
        if self.training and self.condition_dropout:
            drop = torch.rand(len(clean), device=clean.device) < self.condition_dropout
            future_ego = torch.where(drop[:, None, None], torch.zeros_like(future_ego), future_ego)
            future_ego_valid = future_ego_valid & ~drop[:, None, None]
        prediction = self.denoiser(
            mixed,
            known,
            history_mask,
            future_ego,
            future_ego_valid,
            self._model_timesteps(timesteps),
        )
        target = self.scheduler.velocity(clean, noise)
        squared = (prediction - target).square() * future_mask
        denominator = future_mask.sum() * clean.shape[2]
        loss = squared.sum() / denominator.clamp_min(1)
        per_future_latent = squared[:, history_frames:].mean(dim=(0, 2, 3, 4))
        return {
            "loss": loss,
            "flow_loss": loss.detach(),
            "timesteps": timesteps.detach(),
            "per_future_latent_loss": per_future_latent.detach(),
        }

    def forward(
        self,
        past_rgb=None,
        future_rgb=None,
        future_ego=None,
        future_ego_valid=None,
        past_latent=None,
        future_latent=None,
    ):
        if past_latent is not None and future_latent is not None:
            return self.training_loss_latents(
                past_latent, future_latent, future_ego, future_ego_valid
            )
        if past_rgb is None or future_rgb is None:
            raise ValueError("Provide either RGB inputs or V2-compatible cached latent inputs")
        return self.training_loss(past_rgb, future_rgb, future_ego, future_ego_valid)

    def _predict_velocity(
        self,
        latent,
        known,
        history_mask,
        future_ego,
        future_ego_valid,
        timesteps,
        guidance,
    ):
        model_timesteps = self._model_timesteps(timesteps)
        conditional = self.denoiser(
            latent, known, history_mask, future_ego, future_ego_valid, model_timesteps
        )
        if guidance == 1.0:
            return conditional
        unconditional = self.denoiser(
            latent,
            known,
            history_mask,
            torch.zeros_like(future_ego),
            torch.zeros_like(future_ego_valid),
            model_timesteps,
        )
        return unconditional + guidance * (conditional - unconditional)

    @torch.no_grad() if torch is not None else (lambda fn: fn)
    def sample(
        self,
        past_rgb,
        future_ego,
        future_ego_valid,
        num_steps: int = 30,
        guidance: float = 1.0,
        sampler: str | None = None,
    ):
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        sampler = sampler or self.default_sampler
        if sampler not in {"euler", "heun"}:
            raise ValueError(f"Unknown Rectified Flow sampler: {sampler}")
        selected_past = self._select_history_rgb(past_rgb)
        past_latent = self.vae.encode(selected_past)
        batch, _, channels, height, width = past_latent.shape
        future_frames = future_ego.shape[1]
        future_latent_frames = self.vae.latent_frame_count(future_frames)
        future_noise = torch.randn(
            batch,
            future_latent_frames,
            channels,
            height,
            width,
            device=past_latent.device,
            dtype=past_latent.dtype,
        )
        latent = torch.cat([past_latent, future_noise], dim=1)
        history_mask, future_mask = self._masks(latent, past_latent.shape[1])
        known = history_mask * latent
        time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=latent.device)
        for index in range(num_steps):
            current_t = time_grid[index].expand(batch)
            next_t = time_grid[index + 1].expand(batch)
            delta = time_grid[index + 1] - time_grid[index]
            velocity = self._predict_velocity(
                latent,
                known,
                history_mask,
                future_ego,
                future_ego_valid,
                current_t,
                guidance,
            )
            proposal = latent + delta * velocity
            proposal = history_mask * known + future_mask * proposal
            if sampler == "heun" and index + 1 < num_steps:
                next_velocity = self._predict_velocity(
                    proposal,
                    known,
                    history_mask,
                    future_ego,
                    future_ego_valid,
                    next_t,
                    guidance,
                )
                next_latent = latent + delta * 0.5 * (velocity + next_velocity)
                latent = history_mask * known + future_mask * next_latent
            else:
                latent = proposal
        return self.vae.decode(latent[:, past_latent.shape[1] :], output_frames=future_frames)
