from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None

from .video_vae import CogVideoXVAEAdapter


TEMPORAL_ENCODING_PROTOCOL = "diffusers_internal_cache_v1"


class MagicCogVideoXVAEAdapter(CogVideoXVAEAdapter):
    """CogVideoX VAE wrapper with cache-preserving temporal micro-chunking.

    Diffusers already splits a video into eight-frame encoder chunks internally
    and carries its causal-convolution cache between those chunks.  The wrapper
    must therefore pass each complete temporal sample to one ``vae.encode``
    call.  Only the batch dimension is micro-batched here.
    """

    def __init__(
        self,
        pretrained: str,
        subfolder: str | None = "vae",
        local_files_only: bool = True,
        micro_frame_size: int = 8,
        micro_batch_size: int = 1,
        posterior: str = "sample",
    ):
        super().__init__(pretrained, subfolder, local_files_only)
        if micro_frame_size < 1 or micro_batch_size < 1:
            raise ValueError("micro_frame_size and micro_batch_size must be positive")
        if posterior not in {"sample", "mode"}:
            raise ValueError("posterior must be sample or mode")
        self.micro_frame_size = int(micro_frame_size)
        self.micro_batch_size = int(micro_batch_size)
        self.posterior = posterior

    def _posterior_value(self, distribution, generator=None):
        if self.posterior == "mode":
            return distribution.mode()
        if generator is None:
            return distribution.sample()
        try:
            return distribution.sample(generator=generator)
        except TypeError as exc:
            raise TypeError("This VAE posterior does not support an explicit generator") from exc

    def _clear_encode_cache(self):
        clear = getattr(self.vae, "_clear_fake_context_parallel_cache", None)
        if clear is not None:
            clear()

    def _encode_value(self, value, generator=None):
        distribution = self.vae.encode(value).latent_dist
        return self._posterior_value(distribution, generator=generator) * self.scaling_factor

    def _encode_micro_batch(self, video, generator=None):
        frames = video.shape[1]
        valid_chunk_layout = (
            frames <= self.micro_frame_size + 1
            or frames % self.micro_frame_size == 0
            or (frames - 1) % self.micro_frame_size == 0
        )
        if not valid_chunk_layout:
            raise ValueError(
                f"MagicDrive VAE expects T=8n or 8n+1, got {frames} frames"
            )
        # Do not call vae.encode once per temporal chunk.  Public Diffusers
        # encode calls reset conv_cache; one full-video call lets its internal
        # 8-frame loop preserve temporal context across the RGB 8/9 boundary.
        latent = self._encode_value(video.transpose(1, 2), generator=generator)
        self._clear_encode_cache()
        return latent.transpose(1, 2).contiguous()

    def latent_frame_count(self, input_frames: int) -> int:
        """Return the exact MagicDrive latent count for supported RGB layouts."""
        frames = int(input_frames)
        if frames < 1:
            raise ValueError("input_frames must be positive")
        if frames % self.micro_frame_size == 0:
            return frames // self.temporal_compression_ratio
        if (frames - 1) % self.micro_frame_size == 0:
            return (frames - 1) // self.temporal_compression_ratio + 1
        if frames <= self.micro_frame_size + 1:
            return super().latent_frame_count(frames)
        raise ValueError(f"MagicDrive VAE expects T=8n or 8n+1, got {frames} frames")

    def encode(self, video, generator=None):
        if video.ndim != 5:
            raise ValueError("video must use [B,T,C,H,W] layout")
        parameter = next(self.vae.parameters())
        video = video.to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            chunks = [
                self._encode_micro_batch(value, generator=generator)
                for value in video.split(self.micro_batch_size, dim=0)
            ]
        return torch.cat(chunks, dim=0)

    def decode(self, latent, output_frames=None):
        if latent.ndim != 5:
            raise ValueError("latent must use [B,T,C,H,W] layout")
        parameter = next(self.vae.parameters())
        latent = latent.to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            chunks = []
            for value in latent.split(self.micro_batch_size, dim=0):
                decoded = self.vae.decode((value / self.scaling_factor).transpose(1, 2)).sample
                chunks.append(decoded.transpose(1, 2))
            video = torch.cat(chunks, dim=0)
        return video[:, :output_frames] if output_frames is not None else video

    def encode_i2v_training_clip(self, history, future, generator=None):
        if history.ndim != 5 or history.shape[1] < 1:
            raise ValueError("history must be [B,T_history,C,H,W]")
        if future.ndim != 5 or future.shape[1] != 16:
            raise ValueError("Stage-3 I2V adaptation requires 16 future RGB frames")
        if history.shape[0] != future.shape[0] or history.shape[2:] != future.shape[2:]:
            raise ValueError("history and future batch/spatial shapes must match")
        history_latents = self.latent_frame_count(history.shape[1])
        total_frames = history.shape[1] + future.shape[1]
        expected_latents = self.latent_frame_count(total_frames)
        latent = self.encode(torch.cat([history, future], dim=1), generator=generator)
        if latent.shape[1] != expected_latents:
            raise RuntimeError(
                f"Expected {total_frames} RGB frames to produce {expected_latents} latents, "
                f"got {latent.shape[1]}"
            )
        x_mask = torch.ones(latent.shape[0], latent.shape[1], device=latent.device, dtype=torch.bool)
        x_mask[:, :history_latents] = False
        return latent, x_mask

    def encode_history(self, history, generator=None):
        if history.ndim != 5 or history.shape[1] < 1:
            raise ValueError("history must be [B,T_history,C,H,W]")
        expected_latents = self.latent_frame_count(history.shape[1])
        latent = self.encode(history, generator=generator)
        if latent.shape[1] != expected_latents:
            raise RuntimeError(
                f"Expected {history.shape[1]} history frames to produce "
                f"{expected_latents} latents, got {latent.shape[1]}"
            )
        return latent

    def encode_anchor(self, anchor, generator=None):
        """Backward-compatible one-frame history entry point."""
        if anchor.ndim != 5 or anchor.shape[1] != 1:
            raise ValueError("anchor must be [B,1,C,H,W]")
        return self.encode_history(anchor, generator=generator)
