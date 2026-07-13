from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = object


class IdentityVideoVAE(nn.Module if torch is not None else object):
    """Debug-only VAE that leaves RGB unchanged; never use it for full training."""

    latent_channels = 3

    def __init__(self):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()

    def encode(self, video):
        return video

    def decode(self, latent, output_frames=None):
        value = latent.clamp(-1, 1)
        return value[:, :output_frames] if output_frames is not None else value

    def latent_frame_count(self, input_frames: int) -> int:
        return input_frames


class LatentShapeOnlyVAE(nn.Module if torch is not None else object):
    """Parameter-free placeholder for training entirely from cached latents."""

    def __init__(self, latent_channels: int = 16, temporal_compression_ratio: int = 4):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.latent_channels = latent_channels
        self.temporal_compression_ratio = temporal_compression_ratio

    def latent_frame_count(self, input_frames: int) -> int:
        remainder = (input_frames - 1) % self.temporal_compression_ratio
        padded = input_frames + (self.temporal_compression_ratio - remainder if remainder else 0)
        return (padded - 1) // self.temporal_compression_ratio + 1

    def encode(self, _video):
        raise RuntimeError("RGB encoding is unavailable in cached-latent mode")

    def decode(self, _latent, output_frames=None):
        raise RuntimeError("Decoding is unavailable in cached-latent mode")


class CogVideoXVAEAdapter(nn.Module if torch is not None else object):
    """Adapter around diffusers.AutoencoderKLCogVideoX using [B,T,C,H,W]."""

    def __init__(
        self,
        pretrained: str,
        subfolder: str | None = "vae",
        local_files_only: bool = True,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        try:
            from diffusers import AutoencoderKLCogVideoX
        except ImportError as exc:
            raise RuntimeError("Install diffusers with CogVideoX VAE support") from exc
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            pretrained,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )
        self.vae.requires_grad_(False).eval()
        self.scaling_factor = float(self.vae.config.scaling_factor)
        self.latent_channels = int(self.vae.config.latent_channels)
        self.temporal_compression_ratio = int(
            getattr(self.vae.config, "temporal_compression_ratio", 4)
        )

    def train(self, mode: bool = True):
        """Keep the frozen VAE in eval mode when its parent enters train mode.

        ``nn.Module.train()`` is recursive.  Without this override,
        ``MaskedVideoDiffusion.train()`` (including the call after validation)
        silently switches the CogVideoX VAE back to training mode.  MagicDrive
        keeps its VAE outside the trainable model and therefore never has this
        state transition.
        """
        super().train(False)
        self.vae.eval()
        return self

    def _pad_video(self, video):
        frames = video.shape[1]
        remainder = (frames - 1) % self.temporal_compression_ratio
        if remainder:
            padding = self.temporal_compression_ratio - remainder
            video = torch.cat([video, video[:, -1:].expand(-1, padding, -1, -1, -1)], dim=1)
        return video

    def latent_frame_count(self, input_frames: int) -> int:
        padded = input_frames
        remainder = (padded - 1) % self.temporal_compression_ratio
        if remainder:
            padded += self.temporal_compression_ratio - remainder
        return (padded - 1) // self.temporal_compression_ratio + 1

    def encode(self, video):
        with torch.no_grad():
            video = self._pad_video(video)
            value = video.transpose(1, 2)
            # Posterior mode keeps latent caches reproducible across runs.
            latent = self.vae.encode(value).latent_dist.mode() * self.scaling_factor
        return latent.transpose(1, 2).contiguous()

    def decode(self, latent, output_frames=None):
        with torch.no_grad():
            value = (latent / self.scaling_factor).transpose(1, 2)
            video = self.vae.decode(value).sample
        video = video.transpose(1, 2)
        return video[:, :output_frames] if output_frames is not None else video
