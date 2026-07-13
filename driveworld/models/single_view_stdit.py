from __future__ import annotations

import math

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = object
    F = None

from .common import timestep_embedding
from .ego_encoder import EgoTrajectoryEncoder


def _sincos_1d(length: int, dim: int, device, dtype):
    """Deterministic position embedding that works for variable sequence lengths."""
    if dim < 2:
        return torch.zeros(length, dim, device=device, dtype=dtype)
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    values = positions * frequencies[None]
    embedding = torch.cat([values.sin(), values.cos()], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding.to(dtype=dtype)


def _sincos_2d(height: int, width: int, dim: int, device, dtype):
    height_dim = dim // 2
    width_dim = dim - height_dim
    height_embedding = _sincos_1d(height, height_dim, device, dtype)
    width_embedding = _sincos_1d(width, width_dim, device, dtype)
    height_embedding = height_embedding[:, None].expand(height, width, height_dim)
    width_embedding = width_embedding[None].expand(height, width, width_dim)
    return torch.cat([height_embedding, width_embedding], dim=-1).reshape(height * width, dim)


class AlignedEgoConditioner(nn.Module if torch is not None else object):
    """Resample frame-rate Ego signals before encoding them as latent-rate tokens."""

    def __init__(self, ego_dim: int, hidden_size: int):
        super().__init__()
        self.ego_dim = ego_dim
        self.encoder = EgoTrajectoryEncoder(ego_dim, hidden_size)

    def _resample(self, ego, valid, output_frames: int):
        if output_frames < 1:
            raise ValueError("output_frames must be positive")
        if ego.shape[1] == output_frames:
            return ego, valid
        weights = valid.to(dtype=ego.dtype)
        weighted = ego * weights
        numerator = F.interpolate(
            weighted.transpose(1, 2), size=output_frames, mode="linear", align_corners=False
        ).transpose(1, 2)
        denominator = F.interpolate(
            weights.transpose(1, 2), size=output_frames, mode="linear", align_corners=False
        ).transpose(1, 2)
        aligned_valid = denominator > 0.5
        aligned = numerator / denominator.clamp_min(1e-6)
        aligned = torch.where(aligned_valid, aligned, torch.zeros_like(aligned))
        return aligned, aligned_valid

    def forward(self, ego, valid, output_frames: int):
        if ego.ndim != 3 or ego.shape[-1] != self.ego_dim:
            raise ValueError(f"Expected Ego [B,T,{self.ego_dim}], got {tuple(ego.shape)}")
        aligned, aligned_valid = self._resample(ego, valid, output_frames)
        return self.encoder(aligned, aligned_valid), aligned_valid


class FactorizedSTDiTBlock(nn.Module if torch is not None else object):
    """Factorized spatial/temporal attention with per-time AdaLN conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.temporal_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.spatial_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 9))

    @staticmethod
    def _modulate(value, shift, scale):
        return value * (1 + scale) + shift

    def forward(self, tokens, condition):
        # tokens: [B,T,S,D], condition: [B,T,D]
        batch, frames, spatial, hidden = tokens.shape
        modulation = self.modulation(condition).chunk(9, dim=-1)
        shift_s, scale_s, gate_s, shift_t, scale_t, gate_t, shift_m, scale_m, gate_m = modulation

        spatial_input = self._modulate(
            self.spatial_norm(tokens), shift_s[:, :, None], scale_s[:, :, None]
        ).reshape(batch * frames, spatial, hidden)
        spatial_output, _ = self.spatial_attention(
            spatial_input, spatial_input, spatial_input, need_weights=False
        )
        spatial_output = spatial_output.reshape(batch, frames, spatial, hidden)
        tokens = tokens + gate_s[:, :, None] * spatial_output

        temporal_input = self._modulate(
            self.temporal_norm(tokens), shift_t[:, :, None], scale_t[:, :, None]
        )
        temporal_input = temporal_input.permute(0, 2, 1, 3).reshape(
            batch * spatial, frames, hidden
        )
        temporal_output, _ = self.temporal_attention(
            temporal_input, temporal_input, temporal_input, need_weights=False
        )
        temporal_output = temporal_output.reshape(batch, spatial, frames, hidden).permute(0, 2, 1, 3)
        tokens = tokens + gate_t[:, :, None] * temporal_output

        mlp_input = self._modulate(self.mlp_norm(tokens), shift_m[:, :, None], scale_m[:, :, None])
        return tokens + gate_m[:, :, None] * self.mlp(mlp_input)


class SingleViewSTDiT(nn.Module if torch is not None else object):
    """Single-view latent video DiT for masked anchor/history-conditioned prediction."""

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_size: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        ego_dim: int = 9,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        fps: float = 6.0,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        if patch_size[0] != 1:
            raise ValueError("V2 currently requires temporal patch size 1")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.latent_channels = latent_channels
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.fps = float(fps)
        self.input_embedder = nn.Conv3d(
            latent_channels + 1,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.history_embedder = nn.Conv3d(
            latent_channels + 1,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.history_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(depth)
        )
        self.ego_conditioner = AlignedEgoConditioner(ego_dim, hidden_size)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fps_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.blocks = nn.ModuleList(
            FactorizedSTDiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = nn.ConvTranspose3d(
            hidden_size,
            latent_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.gradient_checkpointing = False
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # The extra history branch starts as a no-op, like a ControlNet residual.
        for projection in self.history_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled

    def _run_block(self, block, tokens, condition):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(block, tokens, condition, use_reentrant=False)
        return block(tokens, condition)

    def forward(self, noisy, known_history, history_mask, future_ego, future_ego_valid, timesteps):
        if noisy.ndim != 5:
            raise ValueError("Expected noisy latent [B,T,C,H,W]")
        if noisy.shape != known_history.shape:
            raise ValueError("noisy and known_history must have identical shapes")
        batch, frames, _, height, width = noisy.shape
        history_frames = int(history_mask[0, :, 0, 0, 0].sum().item())
        if history_frames < 1 or history_frames >= frames:
            raise ValueError("Sequence must contain both history and future latent frames")
        future_frames = frames - history_frames
        patch_t, patch_h, patch_w = self.patch_size
        if frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError(
                f"Latent shape {(frames, height, width)} is not divisible by patch {self.patch_size}"
            )

        mask_channel = history_mask.transpose(1, 2)
        input_value = torch.cat([noisy.transpose(1, 2), mask_channel], dim=1)
        history_value = torch.cat([known_history.transpose(1, 2), mask_channel], dim=1)
        embedded = self.input_embedder(input_value)
        history = self.history_embedder(history_value)
        _, hidden, token_t, token_h, token_w = embedded.shape
        spatial_tokens = token_h * token_w
        tokens = embedded.permute(0, 2, 3, 4, 1).reshape(batch, token_t, spatial_tokens, hidden)
        history_tokens = history.permute(0, 2, 3, 4, 1).reshape(
            batch, token_t, spatial_tokens, hidden
        )

        temporal_position = _sincos_1d(token_t, hidden, noisy.device, tokens.dtype)
        spatial_position = _sincos_2d(token_h, token_w, hidden, noisy.device, tokens.dtype)
        tokens = tokens + temporal_position[None, :, None] + spatial_position[None, None]

        ego_tokens, _ = self.ego_conditioner(future_ego, future_ego_valid, future_frames)
        ego_tokens = torch.cat(
            [ego_tokens.new_zeros(batch, history_frames, hidden), ego_tokens], dim=1
        )
        if token_t != frames:
            ego_tokens = F.interpolate(
                ego_tokens.transpose(1, 2), size=token_t, mode="linear", align_corners=False
            ).transpose(1, 2)
        time_tokens = self.timestep_mlp(timestep_embedding(timesteps, hidden)).to(tokens.dtype)
        fps = torch.full((batch,), self.fps, device=noisy.device, dtype=torch.float32)
        fps_tokens = self.fps_mlp(timestep_embedding(fps, hidden)).to(tokens.dtype)
        condition = ego_tokens.to(tokens.dtype) + temporal_position[None] + time_tokens[:, None] + fps_tokens[:, None]

        for block, history_projection in zip(self.blocks, self.history_projections):
            tokens = tokens + history_projection(history_tokens)
            tokens = self._run_block(block, tokens, condition)
        tokens = self.final_norm(tokens)
        tokens = tokens.reshape(batch, token_t, token_h, token_w, hidden).permute(0, 4, 1, 2, 3)
        output = self.output(tokens).transpose(1, 2)
        return output[:, :frames, :, :height, :width]
