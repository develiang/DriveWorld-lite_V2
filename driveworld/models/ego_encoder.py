from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ImportError:  # Allows data-only tooling without a CUDA/PyTorch environment.
    torch = None
    nn = object


class EgoTrajectoryEncoder(nn.Module if torch is not None else object):
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        fourier_bands: int = 4,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required to construct EgoTrajectoryEncoder")
        super().__init__()
        self.input_dim = input_dim
        self.fourier_bands = fourier_bands
        embedded_dim = input_dim * (1 + 2 * fourier_bands) + input_dim
        self.input_projection = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, ego, valid_mask=None):
        if ego.ndim != 3 or ego.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [B,T,{self.input_dim}] ego tensor, got {tuple(ego.shape)}")
        if valid_mask is None:
            valid_mask = torch.ones_like(ego, dtype=torch.bool)
        valid = valid_mask.to(dtype=ego.dtype)
        clean = torch.where(valid_mask, ego, torch.zeros_like(ego))
        frequencies = (2.0 ** torch.arange(self.fourier_bands, device=ego.device)) * math.pi
        phase = clean.unsqueeze(-1) * frequencies
        fourier = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(-2)
        tokens = self.input_projection(torch.cat([clean, fourier, valid], dim=-1))
        frame_missing = ~valid_mask.any(dim=-1)
        tokens = self.transformer(tokens, src_key_padding_mask=frame_missing)
        return self.output_norm(tokens)

