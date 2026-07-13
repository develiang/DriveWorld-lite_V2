from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = object
    F = None


def timestep_embedding(timesteps, dim: int, max_period: int = 10000):
    half = dim // 2
    frequencies = torch.exp(
        -torch.log(torch.tensor(float(max_period), device=timesteps.device))
        * torch.arange(half, device=timesteps.device)
        / half
    )
    values = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(values), torch.sin(values)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConditionedResBlock3D(nn.Module if torch is not None else object):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        groups_in = min(32, in_channels)
        groups_out = min(32, out_channels)
        while in_channels % groups_in:
            groups_in -= 1
        while out_channels % groups_out:
            groups_out -= 1
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, out_channels * 2)
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, condition):
        # condition: [B,T,C], x: [B,C,T,H,W]
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        scale = scale.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        shift = shift.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return self.skip(x) + h


class TemporalCrossConditioner(nn.Module if torch is not None else object):
    def __init__(self, video_channels: int, ego_dim: int, output_dim: int, heads: int = 4):
        super().__init__()
        self.query = nn.Linear(video_channels, output_dim)
        self.ego = nn.Linear(ego_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, video, ego_tokens):
        pooled = video.mean(dim=(-1, -2)).transpose(1, 2)
        attended, _ = self.attention(self.query(pooled), self.ego(ego_tokens), self.ego(ego_tokens))
        return self.norm(attended + self.query(pooled))

