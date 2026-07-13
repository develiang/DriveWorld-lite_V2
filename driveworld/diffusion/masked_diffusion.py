from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = object
    F = None

from .scheduler import LinearNoiseScheduler


class MaskedVideoDiffusion(nn.Module if torch is not None else object):
    def __init__(
        self,
        vae,
        denoiser,
        scheduler: LinearNoiseScheduler,
        history_frames: int = 8,
        condition_dropout: float = 0.1,
        timestep_sampling: str = "uniform",
        low_timestep_fraction: float = 0.5,
        low_timestep_max: int = 250,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.vae = vae
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.history_frames = history_frames
        self.condition_dropout = condition_dropout
        self.timestep_sampling = timestep_sampling
        self.low_timestep_fraction = low_timestep_fraction
        self.low_timestep_max = low_timestep_max

    def _sample_timesteps(self, batch_size: int, device):
        timesteps = torch.randint(self.scheduler.num_train_timesteps, (batch_size,), device=device)
        if self.timestep_sampling == "uniform":
            return timesteps
        if self.timestep_sampling == "mixed_low":
            select_low = torch.rand(batch_size, device=device) < self.low_timestep_fraction
            low = torch.randint(
                min(self.low_timestep_max, self.scheduler.num_train_timesteps),
                (batch_size,),
                device=device,
            )
            return torch.where(select_low, low, timesteps)
        raise ValueError(f"Unknown timestep sampling strategy: {self.timestep_sampling}")

    def _masks(self, latent, history_latent_frames: int):
        batch, frames, _, height, width = latent.shape
        history = latent.new_zeros(batch, frames, 1, height, width)
        history[:, :history_latent_frames] = 1
        return history, 1 - history

    def training_loss(self, past_rgb, future_rgb, future_ego, future_ego_valid):
        with torch.inference_mode():
            past_clean = self.vae.encode(past_rgb)
            future_clean = self.vae.encode(future_rgb)
        # inference_mode tensors cannot be saved for backward by the denoiser.
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
        clean = torch.cat([past_clean, future_clean], dim=1)
        history_latent_frames = past_clean.shape[1]
        history_mask, future_mask = self._masks(clean, history_latent_frames)
        if timesteps is None:
            timesteps = self._sample_timesteps(len(clean), clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        noisy_all = self.scheduler.add_noise(clean, noise, timesteps)
        mixed = history_mask * clean + future_mask * noisy_all
        known = history_mask * clean
        if self.training and self.condition_dropout:
            drop = torch.rand(len(clean), device=clean.device) < self.condition_dropout
            future_ego = torch.where(drop[:, None, None], torch.zeros_like(future_ego), future_ego)
            future_ego_valid = future_ego_valid & ~drop[:, None, None]
        prediction = self.denoiser(
            mixed, known, history_mask, future_ego, future_ego_valid, timesteps
        )
        target = self.scheduler.velocity(clean, noise, timesteps)
        squared = (prediction - target).square() * future_mask
        denominator = future_mask.sum() * clean.shape[2]
        loss = squared.sum() / denominator.clamp_min(1)
        return {"loss": loss, "diffusion_loss": loss.detach(), "timesteps": timesteps}

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
            raise ValueError("Provide either RGB inputs or cached latent inputs")
        return self.training_loss(past_rgb, future_rgb, future_ego, future_ego_valid)

    @torch.no_grad() if torch is not None else (lambda fn: fn)
    def sample(
        self,
        past_rgb,
        future_ego,
        future_ego_valid,
        num_steps: int = 50,
        guidance=1.0,
        sampler: str = "ddim",
    ):
        past_latent = self.vae.encode(past_rgb)
        batch, _, channels, height, width = past_latent.shape
        future_frames = future_ego.shape[1]
        future_latent_frames = self.vae.latent_frame_count(future_frames)
        latent = torch.randn(
            batch,
            past_latent.shape[1] + future_latent_frames,
            channels,
            height,
            width,
            device=past_latent.device,
            dtype=past_latent.dtype,
        )
        latent[:, : past_latent.shape[1]] = past_latent
        history_mask, future_mask = self._masks(latent, past_latent.shape[1])
        known = history_mask * latent
        if sampler == "ddim":
            try:
                from diffusers import DDIMScheduler
            except ImportError as exc:
                raise RuntimeError("Diffusers is required for the DDIM inference sampler") from exc
            inference_scheduler = DDIMScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="v_prediction",
                clip_sample=False,
                set_alpha_to_one=True,
                timestep_spacing="leading",
            )
            inference_scheduler.set_timesteps(num_steps, device=latent.device)
            step_indices = inference_scheduler.timesteps
        elif sampler == "legacy":
            inference_scheduler = None
            step_indices = torch.linspace(
                self.scheduler.num_train_timesteps - 1, 0, num_steps, device=latent.device
            ).long()
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        for step_index, step in enumerate(step_indices):
            timestep = torch.full((batch,), step, device=latent.device, dtype=torch.long)
            conditional = self.denoiser(
                latent, known, history_mask, future_ego, future_ego_valid, timestep
            )
            if guidance != 1.0:
                unconditional = self.denoiser(
                    latent,
                    known,
                    history_mask,
                    torch.zeros_like(future_ego),
                    torch.zeros_like(future_ego_valid),
                    timestep,
                )
                velocity = unconditional + guidance * (conditional - unconditional)
            else:
                velocity = conditional
            if inference_scheduler is not None:
                next_latent = inference_scheduler.step(
                    velocity, step, latent, eta=0.0, return_dict=True
                ).prev_sample
            else:
                clean = self.scheduler.clean_from_velocity(latent, velocity, timestep)
                if step_index + 1 == len(step_indices):
                    next_latent = clean
                else:
                    next_t = step_indices[step_index + 1].expand(batch)
                    predicted_noise = self.scheduler.noise_from_velocity(latent, velocity, timestep)
                    next_latent = self.scheduler.add_noise(clean, predicted_noise, next_t)
            latent = history_mask * known + future_mask * next_latent
        return self.vae.decode(latent[:, past_latent.shape[1] :], output_frames=future_frames)
