from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = object
    F = None

from .common import ConditionedResBlock3D, TemporalCrossConditioner, timestep_embedding
from .ego_encoder import EgoTrajectoryEncoder


class LatentVideoUNet(nn.Module if torch is not None else object):
    """Small masked-video denoiser with temporal cross-attention and AdaFiLM blocks."""

    def __init__(
        self,
        latent_channels: int = 3,
        base_channels: int = 64,
        ego_dim: int = 9,
        condition_dim: int = 256,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.latent_channels = latent_channels
        self.ego_encoder = EgoTrajectoryEncoder(ego_dim, condition_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 4),
            nn.SiLU(),
            nn.Linear(condition_dim * 4, condition_dim),
        )
        self.input = nn.Conv3d(latent_channels * 2 + 1, base_channels, 3, padding=1)
        self.cross = TemporalCrossConditioner(base_channels, condition_dim, condition_dim)
        self.block1 = ConditionedResBlock3D(base_channels, base_channels, condition_dim)
        self.down = nn.Conv3d(base_channels, base_channels * 2, (3, 4, 4), (1, 2, 2), (1, 1, 1))
        self.block2 = ConditionedResBlock3D(base_channels * 2, base_channels * 2, condition_dim)
        self.mid = ConditionedResBlock3D(base_channels * 2, base_channels * 2, condition_dim)
        self.up = nn.ConvTranspose3d(base_channels * 2, base_channels, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        self.output_block = ConditionedResBlock3D(base_channels * 2, base_channels, condition_dim)
        self.output = nn.Conv3d(base_channels, latent_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled

    def _block(self, block, x, condition):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(block, x, condition, use_reentrant=False)
        return block(x, condition)

    def forward(self, noisy, known_history, history_mask, future_ego, future_ego_valid, timesteps):
        if noisy.shape != known_history.shape:
            raise ValueError("noisy and known_history must have the same shape")
        if history_mask.shape[:2] != noisy.shape[:2]:
            raise ValueError("history_mask must have shape [B,T,1,H,W] or broadcastable equivalent")
        x = torch.cat([noisy, known_history, history_mask], dim=2).transpose(1, 2)
        x = self.input(x)
        ego = self.ego_encoder(future_ego, future_ego_valid)
        history_frames = int(history_mask[0, :, 0, 0, 0].sum().item())
        future_latent_frames = noisy.shape[1] - history_frames
        if ego.shape[1] != future_latent_frames:
            ego = F.interpolate(
                ego.transpose(1, 2), size=future_latent_frames, mode="linear", align_corners=False
            ).transpose(1, 2)
        ego = torch.cat(
            [torch.zeros(ego.shape[0], history_frames, ego.shape[2], device=ego.device, dtype=ego.dtype), ego],
            dim=1,
        )
        time = self.timestep_mlp(timestep_embedding(timesteps, ego.shape[-1]))[:, None]
        condition = self.cross(x, ego) + ego + time
        e = self._block(self.block1, x, condition)
        h = self._block(self.block2, self.down(e), condition)
        h = self._block(self.mid, h, condition)
        h = self.up(h)
        h = self._block(self.output_block, torch.cat([h, e], dim=1), condition)
        return self.output(h).transpose(1, 2)
