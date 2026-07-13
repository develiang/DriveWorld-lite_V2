from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = object


def _as_batch_tensor(value, reference):
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=torch.float32)
    return torch.full_like(reference, float(value), dtype=torch.float32)


def magic_timestep_transform(
    timesteps,
    *,
    height,
    width,
    num_frames,
    num_timesteps: int = 1000,
    base_resolution: int = 512 * 512,
    base_num_frames: int = 1,
    scale: float = 1.0,
    cog_style: bool = True,
):
    """MagicDrive Stage-3 resolution/time shift, including CogVideoX 17 -> 5 mapping."""
    if torch is None:
        raise RuntimeError("PyTorch is required")
    timesteps = timesteps.to(dtype=torch.float32)
    height = _as_batch_tensor(height, timesteps)
    width = _as_batch_tensor(width, timesteps)
    num_frames = _as_batch_tensor(num_frames, timesteps)

    normalized = timesteps / float(num_timesteps)
    ratio_space = ((height * width) / float(base_resolution)).sqrt()
    single_frame = num_frames == 1
    if cog_style:
        latent_frames = torch.div(num_frames, 4, rounding_mode="floor") + num_frames.remainder(2)
    else:
        latent_frames = torch.div(num_frames, 17, rounding_mode="floor") * 5
    latent_frames = torch.where(single_frame, torch.ones_like(latent_frames), latent_frames)
    if not bool((latent_frames >= 1).all()):
        raise ValueError("num_frames maps to fewer than one latent frame")
    ratio_time = (latent_frames / float(base_num_frames)).sqrt()
    ratio = ratio_space * ratio_time * float(scale)
    if not bool((ratio > 0).all()):
        raise ValueError("timestep transform ratio must be positive")
    shifted = ratio * normalized / (1 + (ratio - 1) * normalized)
    return shifted * float(num_timesteps)


class MagicRectifiedFlowScheduler(nn.Module if torch is not None else object):
    """Stage-3 RF convention: t=0 clean, t=num_timesteps noise, target=clean-noise."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        sample_method: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        use_timestep_transform: bool = True,
        transform_scale: float = 1.0,
        cog_style_transform: bool = True,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be positive")
        if sample_method not in {"uniform", "logit_normal"}:
            raise ValueError(f"Unknown Magic RF timestep method: {sample_method}")
        self.num_timesteps = int(num_timesteps)
        self.sample_method = sample_method
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.use_timestep_transform = bool(use_timestep_transform)
        self.transform_scale = float(transform_scale)
        self.cog_style_transform = bool(cog_style_transform)

    def sample_timesteps(self, batch_size: int, device, model_kwargs=None):
        if self.sample_method == "uniform":
            timesteps = torch.rand(batch_size, device=device) * self.num_timesteps
        else:
            logits = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            timesteps = logits.sigmoid() * self.num_timesteps
        if self.use_timestep_transform:
            if model_kwargs is None:
                raise ValueError("height, width and num_frames are required for timestep transform")
            timesteps = self.transform_timesteps(timesteps, model_kwargs)
        return timesteps

    def transform_timesteps(self, timesteps, model_kwargs):
        missing = {"height", "width", "num_frames"} - set(model_kwargs)
        if missing:
            raise ValueError(f"Missing timestep transform inputs: {sorted(missing)}")
        return magic_timestep_transform(
            timesteps,
            height=model_kwargs["height"],
            width=model_kwargs["width"],
            num_frames=model_kwargs["num_frames"],
            num_timesteps=self.num_timesteps,
            scale=self.transform_scale,
            cog_style=self.cog_style_transform,
        )

    @staticmethod
    def _coefficient(timesteps, ndim: int, num_timesteps: int):
        shape = (len(timesteps),) + (1,) * (ndim - 1)
        return (1 - timesteps.float() / float(num_timesteps)).view(shape)

    def add_noise(self, clean, noise, timesteps):
        if clean.shape != noise.shape:
            raise ValueError("clean and noise must have identical shapes")
        clean_weight = self._coefficient(timesteps, clean.ndim, self.num_timesteps).to(
            device=clean.device, dtype=clean.dtype
        )
        return clean_weight * clean + (1 - clean_weight) * noise

    @staticmethod
    def velocity(clean, noise):
        return clean - noise

    def prepare_training_input(self, clean, timesteps, noise=None, x_mask=None):
        """Apply Stage-3 noise and keep x_mask=False temporal positions at t=0."""
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = self.add_noise(clean, noise, timesteps)
        if x_mask is not None:
            if x_mask.shape != (clean.shape[0], clean.shape[2]):
                raise ValueError(
                    f"x_mask must be [B,T]={clean.shape[0], clean.shape[2]}, got {tuple(x_mask.shape)}"
                )
            clean_at_t0 = self.add_noise(clean, noise, torch.zeros_like(timesteps))
            noisy = torch.where(x_mask[:, None, :, None, None], noisy, clean_at_t0)
        return noisy, self.velocity(clean, noise), noise

    @staticmethod
    def masked_mse(prediction, target, x_mask=None):
        squared = (prediction - target).square()
        if x_mask is None:
            return squared.flatten(1).mean(1)
        if squared.ndim != 5 or x_mask.shape != (squared.shape[0], squared.shape[2]):
            raise ValueError("masked_mse expects prediction [B,C,T,H,W] and x_mask [B,T]")
        flattened = squared.permute(0, 2, 1, 3, 4).flatten(2)
        denominator = x_mask.sum(dim=1) * flattened.shape[-1]
        if bool((denominator == 0).any()):
            raise ValueError("Each sample must contain at least one generated frame")
        return (flattened * x_mask[:, :, None]).sum(dim=(1, 2)) / denominator

    def sampling_timesteps(self, batch_size: int, num_steps: int, device, model_kwargs=None):
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        values = [
            torch.full(
                (batch_size,),
                (1.0 - index / num_steps) * self.num_timesteps,
                device=device,
            )
            for index in range(num_steps)
        ]
        if self.use_timestep_transform:
            if model_kwargs is None:
                raise ValueError("model_kwargs are required for timestep transform")
            values = [self.transform_timesteps(value, model_kwargs) for value in values]
        return values
