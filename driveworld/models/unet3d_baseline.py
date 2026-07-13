from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = object
    F = None

from .common import ConditionedResBlock3D
from .ego_encoder import EgoTrajectoryEncoder


class UNet3DBaseline(nn.Module if torch is not None else object):
    """Compact deterministic baseline with future Ego FiLM conditioning."""

    def __init__(
        self,
        base_channels: int = 32,
        ego_dim: int = 9,
        ego_hidden_dim: int = 128,
        future_frames: int = 16,
        use_ego: bool = True,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required to construct UNet3DBaseline")
        super().__init__()
        self.future_frames = future_frames
        self.use_ego = use_ego
        self.ego_encoder = EgoTrajectoryEncoder(ego_dim, ego_hidden_dim)
        self.stem = nn.Conv3d(3, base_channels, 3, padding=1)
        self.enc1 = ConditionedResBlock3D(base_channels, base_channels, ego_hidden_dim)
        self.down1 = nn.Conv3d(base_channels, base_channels * 2, (3, 4, 4), (1, 2, 2), (1, 1, 1))
        self.enc2 = ConditionedResBlock3D(base_channels * 2, base_channels * 2, ego_hidden_dim)
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 4, (3, 4, 4), (1, 2, 2), (1, 1, 1))
        self.mid = ConditionedResBlock3D(base_channels * 4, base_channels * 4, ego_hidden_dim)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        self.dec2 = ConditionedResBlock3D(base_channels * 4, base_channels * 2, ego_hidden_dim)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, (1, 4, 4), (1, 2, 2), (0, 1, 1))
        self.dec1 = ConditionedResBlock3D(base_channels * 2, base_channels, ego_hidden_dim)
        self.output = nn.Conv3d(base_channels, 3, 3, padding=1)

    def forward(self, past_rgb, future_ego, future_ego_valid=None):
        if past_rgb.ndim != 5 or past_rgb.shape[2] != 3:
            raise ValueError("past_rgb must have shape [B,T,3,H,W]")
        if future_ego.shape[1] != self.future_frames:
            raise ValueError(f"Expected {self.future_frames} future Ego frames")
        condition = self.ego_encoder(future_ego, future_ego_valid)
        if not self.use_ego:
            condition = torch.zeros_like(condition)
        x = past_rgb.transpose(1, 2)
        x = F.interpolate(x, size=(self.future_frames, x.shape[-2], x.shape[-1]), mode="trilinear")
        x = self.stem(x)
        e1 = self.enc1(x, condition)
        e2 = self.enc2(self.down1(e1), condition)
        mid = self.mid(self.down2(e2), condition)
        d2 = self.up2(mid)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), condition)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), condition)
        return torch.tanh(self.output(d1)).transpose(1, 2)

