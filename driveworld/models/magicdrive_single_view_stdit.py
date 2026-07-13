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


class MagicRMSNorm(nn.Module if torch is not None else object):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, value):
        dtype = value.dtype
        variance = value.float().pow(2).mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * value.to(dtype)


class MagicRotaryEmbedding(nn.Module if torch is not None else object):
    """Minimal rotary-embedding-torch compatible module for Stage-3 temporal attention."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", frequencies)

    @staticmethod
    def _rotate_half(value):
        value = value.reshape(*value.shape[:-1], -1, 2)
        first, second = value.unbind(dim=-1)
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def rotate_queries_or_keys(self, value):
        sequence = value.shape[-2]
        positions = torch.arange(sequence, device=value.device, dtype=self.freqs.dtype)
        angles = torch.einsum("t,f->tf", positions, self.freqs).repeat_interleave(2, dim=-1)
        angles = angles.to(dtype=value.dtype)
        while angles.ndim < value.ndim:
            angles = angles.unsqueeze(0)
        return value * angles.cos() + self._rotate_half(value) * angles.sin()


class MagicPatchEmbed3D(nn.Module if torch is not None else object):
    def __init__(self, patch_size, in_channels: int, hidden_size: int):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, value):
        _, _, frames, height, width = value.shape
        pad_t = (-frames) % self.patch_size[0]
        pad_h = (-height) % self.patch_size[1]
        pad_w = (-width) % self.patch_size[2]
        if pad_t or pad_h or pad_w:
            value = F.pad(value, (0, pad_w, 0, pad_h, 0, pad_t))
        return self.proj(value).flatten(2).transpose(1, 2)


class MagicTimestepEmbedder(nn.Module if torch is not None else object):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(timesteps, dim: int, max_period: int = 10000):
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, device=timesteps.device).float() / half
        )
        arguments = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps, dtype):
        value = self.timestep_embedding(timesteps, self.frequency_embedding_size).to(dtype)
        return self.mlp(value)


class MagicSizeEmbedder(MagicTimestepEmbedder):
    def forward(self, values, batch_size: int):
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("size values must be [B,D]")
        if values.shape[0] != batch_size:
            if batch_size % values.shape[0]:
                raise ValueError("size batch cannot be repeated to model batch")
            values = values.repeat(batch_size // values.shape[0], 1)
        batch, dimensions = values.shape
        embedded = self.timestep_embedding(
            values.reshape(-1), self.frequency_embedding_size
        ).to(self.mlp[0].weight.dtype)
        return self.mlp(embedded).reshape(batch, dimensions * self.mlp[-1].out_features)


class MagicMLP(nn.Module if torch is not None else object):
    def __init__(self, hidden_size: int, mlp_ratio: float):
        super().__init__()
        intermediate = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, intermediate)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(intermediate, hidden_size)

    def forward(self, value):
        return self.fc2(self.act(self.fc1(value)))


class MagicSelfAttention(nn.Module if torch is not None else object):
    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True, qk_norm=True):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.q_norm = MagicRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = MagicRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, value, rotary=None):
        batch, sequence, hidden = value.shape
        qkv = self.qkv(value).reshape(
            batch, sequence, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, attention_value = qkv.unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        if rotary is not None:
            query, key = rotary(query), rotary(key)
        output = F.scaled_dot_product_attention(
            query,
            key,
            attention_value,
            dropout_p=0.0,
            scale=self.scale,
        )
        output = output.transpose(1, 2).reshape(batch, sequence, hidden)
        return self.proj(output)


class MagicCrossAttention(nn.Module if torch is not None else object):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.kv_linear = nn.Linear(hidden_size, hidden_size * 2)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, value, condition):
        batch, sequence, hidden = value.shape
        condition_sequence = condition.shape[1]
        query = self.q_linear(value).reshape(
            batch, sequence, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key_value = self.kv_linear(condition).reshape(
            batch, condition_sequence, 2, self.num_heads, self.head_dim
        )
        key, attention_value = key_value.unbind(2)
        key = key.permute(0, 2, 1, 3)
        attention_value = attention_value.permute(0, 2, 1, 3)
        output = F.scaled_dot_product_attention(
            query,
            key,
            attention_value,
            dropout_p=0.0,
            scale=self.scale,
        )
        output = output.transpose(1, 2).reshape(
            batch, sequence, hidden
        )
        return self.proj(output)


def _modulate(value, shift, scale):
    return value * (1 + scale) + shift


def _temporal_select(mask, primary, clean, frames: int, spatial: int):
    shape = (primary.shape[0], frames, spatial, primary.shape[-1])
    primary = primary.reshape(shape)
    clean = clean.reshape(shape)
    return torch.where(mask[:, :, None, None], primary, clean).reshape(
        shape[0], frames * spatial, shape[-1]
    )


class MagicSingleViewSTDiTBlock(nn.Module if torch is not None else object):
    """MagicDrive block with the cross-view branch removed at construction time."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        temporal: bool = False,
        is_control_block: bool = False,
    ):
        super().__init__()
        self.temporal = temporal
        self.is_control_block = is_control_block
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = MagicSelfAttention(hidden_size, num_heads, qkv_bias=True, qk_norm=True)
        self.cross_attn = MagicCrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = MagicMLP(hidden_size, mlp_ratio)
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )
        self.after_proj = nn.Linear(hidden_size, hidden_size) if is_control_block else None

    def forward(
        self,
        value,
        condition,
        timestep_modulation,
        *,
        frames: int,
        spatial: int,
        x_mask=None,
        clean_timestep_modulation=None,
        rotary=None,
    ):
        batch = value.shape[0]
        modulation = (
            self.scale_shift_table[None] + timestep_modulation.reshape(batch, 6, -1)
        ).chunk(6, dim=1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation
        clean_modulation = None
        if x_mask is not None:
            if clean_timestep_modulation is None:
                raise ValueError("clean timestep modulation is required with x_mask")
            clean_modulation = (
                self.scale_shift_table[None]
                + clean_timestep_modulation.reshape(batch, 6, -1)
            ).chunk(6, dim=1)

        normalized = _modulate(self.norm1(value), shift_attn, scale_attn)
        if clean_modulation is not None:
            clean_normalized = _modulate(
                self.norm1(value), clean_modulation[0], clean_modulation[1]
            )
            normalized = _temporal_select(
                x_mask, normalized, clean_normalized, frames, spatial
            )
        if self.temporal:
            normalized = normalized.reshape(batch, frames, spatial, -1).permute(0, 2, 1, 3)
            normalized = normalized.reshape(batch * spatial, frames, -1)
            attended = self.attn(normalized, rotary=rotary)
            attended = attended.reshape(batch, spatial, frames, -1).permute(0, 2, 1, 3)
            attended = attended.reshape(batch, frames * spatial, -1)
        else:
            normalized = normalized.reshape(batch, frames, spatial, -1)
            attended = self.attn(normalized.reshape(batch * frames, spatial, -1))
            attended = attended.reshape(batch, frames * spatial, -1)

        gated = gate_attn * attended
        if clean_modulation is not None:
            gated = _temporal_select(
                x_mask,
                gated,
                clean_modulation[2] * attended,
                frames,
                spatial,
            )
        value = value + gated

        if condition.shape[1] == 1:
            value = value + self.cross_attn(value, condition[:, 0])
        elif condition.shape[1] == frames:
            query = value.reshape(batch, frames, spatial, -1).reshape(
                batch * frames, spatial, -1
            )
            cond = condition.reshape(batch * frames, condition.shape[2], -1)
            crossed = self.cross_attn(query, cond)
            value = value + crossed.reshape(batch, frames * spatial, -1)
        else:
            raise ValueError("condition time dimension must be 1 or latent frame count")

        normalized = _modulate(self.norm2(value), shift_mlp, scale_mlp)
        if clean_modulation is not None:
            clean_normalized = _modulate(
                self.norm2(value), clean_modulation[3], clean_modulation[4]
            )
            normalized = _temporal_select(
                x_mask, normalized, clean_normalized, frames, spatial
            )
        mlp_value = self.mlp(normalized)
        gated = gate_mlp * mlp_value
        if clean_modulation is not None:
            gated = _temporal_select(
                x_mask,
                gated,
                clean_modulation[5] * mlp_value,
                frames,
                spatial,
            )
        value = value + gated
        if self.after_proj is not None:
            return value, self.after_proj(value)
        return value


class MagicMapControlEmbedding(nn.Module if torch is not None else object):
    """Stage-3 8-channel BEV encoder with checkpoint-identical parameter names."""

    def __init__(self, hidden_size: int = 1152, map_channels: int = 8):
        super().__init__()
        block_out_channels = (16, 32, 96, 256)
        self.conv_in = nn.Conv2d(map_channels, block_out_channels[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        for index in range(len(block_out_channels) - 2):
            channel_in = block_out_channels[index]
            channel_out = block_out_channels[index + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, 3, padding=1))
            self.blocks.append(
                nn.Conv2d(channel_in, channel_out, 3, stride=2, padding=(2, 1))
            )
        channel_in, channel_out = block_out_channels[-2:]
        self.blocks.append(nn.Conv2d(channel_in, channel_in, 3, padding=(2, 1)))
        self.blocks.append(
            nn.Conv2d(channel_in, channel_out, 3, stride=(2, 1), padding=(2, 1))
        )
        self.conv_out = nn.Conv2d(channel_out, hidden_size // 2, 3, padding=1)

    def forward(self, value):
        value = F.silu(self.conv_in(value))
        for block in self.blocks:
            value = F.silu(block(value))
        return self.conv_out(value)


class MagicCogVideoXDownsample3D(nn.Module if torch is not None else object):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=0)

    @staticmethod
    def _downsample_time(value):
        batch, channels, frames, height, width = value.shape
        value = value.permute(0, 2, 3, 4, 1).reshape(
            batch, frames, height * width, channels
        )
        value = value.permute(0, 2, 3, 1).reshape(batch * height * width, channels, frames)
        if frames % 2:
            first, rest = value[..., :1], value[..., 1:]
            rest = F.avg_pool1d(rest, 2, 2) if rest.shape[-1] else rest
            value = torch.cat([first, rest], dim=-1)
        else:
            value = F.avg_pool1d(value, 2, 2)
        frames = value.shape[-1]
        return value.reshape(batch, height * width, channels, frames).permute(0, 3, 1, 2).reshape(
            batch, frames, height, width, channels
        ).permute(0, 4, 1, 2, 3)

    def forward(self, value):
        value = self._downsample_time(value)
        value = F.pad(value, (0, 1, 0, 1))
        batch, channels, frames, height, width = value.shape
        value = value.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        value = self.conv(value)
        return value.reshape(batch, frames, -1, value.shape[-2], value.shape[-1]).permute(
            0, 2, 1, 3, 4
        )


class MagicMapControlTempEmbedding(nn.Module if torch is not None else object):
    def __init__(self, hidden_size: int = 1152):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            MagicCogVideoXDownsample3D(hidden_size // 2, hidden_size // 2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            MagicCogVideoXDownsample3D(hidden_size // 2, hidden_size),
        )

    def forward(self, value):
        return self.conv_blocks(value)


class MagicPositionEmbedding2D(nn.Module if torch is not None else object):
    def __init__(self, hidden_size: int):
        super().__init__()
        if hidden_size % 4:
            raise ValueError("hidden_size must be divisible by four")
        half = hidden_size // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _sincos(self, positions):
        value = torch.einsum("i,d->id", positions, self.inv_freq)
        return torch.cat((value.sin(), value.cos()), dim=-1)

    def forward(self, reference, height: int, width: int, scale=1.0, base_size=None):
        grid_h = torch.arange(height, device=reference.device) / scale
        grid_w = torch.arange(width, device=reference.device) / scale
        if base_size is not None:
            grid_h *= base_size / height
            grid_w *= base_size / width
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        return torch.cat([self._sincos(grid_h), self._sincos(grid_w)], dim=-1)[None].to(
            reference.dtype
        )


class MagicFinalLayer(nn.Module if torch is not None else object):
    def __init__(self, hidden_size: int, patch_volume: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_volume * out_channels)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )

    def forward(self, value, timestep, frames, spatial, x_mask=None, clean_timestep=None):
        shift, scale = (self.scale_shift_table[None] + timestep[:, None]).chunk(2, dim=1)
        value = _modulate(self.norm_final(value), shift, scale)
        if x_mask is not None:
            shift_clean, scale_clean = (
                self.scale_shift_table[None] + clean_timestep[:, None]
            ).chunk(2, dim=1)
            clean_value = _modulate(self.norm_final(value), shift_clean, scale_clean)
            value = _temporal_select(x_mask, value, clean_value, frames, spatial)
        return self.linear(value)


class MagicDriveSingleViewSTDiT(nn.Module if torch is not None else object):
    """Base-only, NC=1 MDDiT retaining Stage-3 spatial/temporal parameter names."""

    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size=(1, 2, 2),
        input_sq_size: int = 512,
        control_depth: int = 0,
        map_channels: int = 8,
        zero_map_size: int = 200,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.input_sq_size = input_sq_size
        if not 0 <= control_depth <= depth:
            raise ValueError("control_depth must be between zero and depth")
        self.control_depth = int(control_depth)
        self.map_channels = int(map_channels)
        self.zero_map_size = int(zero_map_size)
        self.pos_embed = MagicPositionEmbedding2D(hidden_size)
        self.rope = MagicRotaryEmbedding(hidden_size // num_heads)
        self.x_embedder = MagicPatchEmbed3D(self.patch_size, in_channels, hidden_size)
        self.t_embedder = MagicTimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.fps_embedder = MagicSizeEmbedder(hidden_size)
        self.register_buffer("base_token", torch.randn(hidden_size))
        self.base_blocks_s = nn.ModuleList(
            MagicSingleViewSTDiTBlock(hidden_size, num_heads, mlp_ratio, temporal=False)
            for _ in range(depth)
        )
        self.base_blocks_t = nn.ModuleList(
            MagicSingleViewSTDiTBlock(hidden_size, num_heads, mlp_ratio, temporal=True)
            for _ in range(depth)
        )
        if self.control_depth:
            self.x_control_embedder = MagicPatchEmbed3D(
                self.patch_size, in_channels, hidden_size
            )
            self.controlnet_cond_embedder = MagicMapControlEmbedding(
                hidden_size, map_channels
            )
            self.controlnet_cond_embedder_temp = MagicMapControlTempEmbedding(hidden_size)
            self.controlnet_cond_patchifier = MagicPatchEmbed3D(
                self.patch_size, hidden_size, hidden_size
            )
            self.before_proj = nn.Linear(hidden_size, hidden_size)
            self.control_blocks_s = nn.ModuleList(
                MagicSingleViewSTDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    temporal=False,
                    is_control_block=True,
                )
                for _ in range(self.control_depth)
            )
            self.control_blocks_t = nn.ModuleList(
                MagicSingleViewSTDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    temporal=True,
                    is_control_block=True,
                )
                for _ in range(self.control_depth)
            )
        self.final_layer = MagicFinalLayer(
            hidden_size, math.prod(self.patch_size), self.out_channels
        )
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = bool(enabled)

    def _run_block(self, block, *args, **kwargs):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            def forward(*values):
                return block(*values, **kwargs)

            return checkpoint(forward, *args, use_reentrant=False)
        return block(*args, **kwargs)

    def _unpatchify(self, value, frames, height, width, real_shape):
        patch_t, patch_h, patch_w = self.patch_size
        batch = value.shape[0]
        value = value.reshape(
            batch,
            frames,
            height,
            width,
            patch_t,
            patch_h,
            patch_w,
            self.out_channels,
        )
        value = value.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            batch,
            self.out_channels,
            frames * patch_t,
            height * patch_h,
            width * patch_w,
        )
        return value[:, :, : real_shape[0], : real_shape[1], : real_shape[2]]

    def _encode_control_map(self, maps, latent, frames, token_h, token_w):
        batch = latent.shape[0]
        if maps is None:
            maps = torch.zeros(
                batch,
                17,
                self.map_channels,
                self.zero_map_size,
                self.zero_map_size,
                device=latent.device,
                dtype=latent.dtype,
            )
        elif maps.ndim == 4:
            maps = maps[:, None].expand(-1, 17, -1, -1, -1)
        if maps.ndim != 5 or maps.shape[:3] != (batch, 17, self.map_channels):
            raise ValueError(
                f"static_maps must be [B,8,H,W] or [B,17,8,H,W], got {tuple(maps.shape)}"
            )
        maps = maps.to(device=latent.device, dtype=latent.dtype)
        map_height, map_width = maps.shape[-2:]
        maps = maps.reshape(batch * 17, self.map_channels, map_height, map_width)
        maps = self.controlnet_cond_embedder(maps)
        maps = maps.reshape(batch, 17, maps.shape[1], maps.shape[2], maps.shape[3]).permute(
            0, 2, 1, 3, 4
        )
        maps = self.controlnet_cond_embedder_temp(maps)
        target_shape = (frames, token_h * self.patch_size[1], token_w * self.patch_size[2])
        if maps.shape[-3:] != target_shape:
            maps = F.interpolate(maps, target_shape)
        return self.controlnet_cond_patchifier(maps).reshape(
            batch, frames, token_h * token_w, -1
        )

    def forward(
        self,
        latent,
        timesteps,
        condition_tokens,
        *,
        fps,
        height,
        width,
        x_mask=None,
        static_maps=None,
    ):
        if latent.ndim != 5:
            raise ValueError("latent must be [B,C,T,H,W]")
        batch, _, real_t, real_h, real_w = latent.shape
        patch_t, patch_h, patch_w = self.patch_size
        frames = math.ceil(real_t / patch_t)
        token_h = math.ceil(real_h / patch_h)
        token_w = math.ceil(real_w / patch_w)
        spatial = token_h * token_w
        dtype = self.x_embedder.proj.weight.dtype
        latent = latent.to(dtype)
        timesteps = timesteps.to(dtype)

        resolution = (float(height) * float(width)) ** 0.5
        scale = resolution / self.input_sq_size
        position = self.pos_embed(
            latent,
            token_h,
            token_w,
            scale=scale,
            base_size=round(spatial**0.5),
        )
        timestep = self.t_embedder(timesteps, dtype)
        fps_tensor = torch.as_tensor(fps, device=latent.device, dtype=dtype)
        if fps_tensor.ndim == 0:
            fps_tensor = fps_tensor.expand(batch)
        timestep = timestep + self.fps_embedder(fps_tensor[:, None], batch)
        timestep_modulation = self.t_block(timestep)
        clean_timestep = None
        clean_modulation = None
        if x_mask is not None:
            clean_timestep = self.t_embedder(torch.zeros_like(timesteps), dtype)
            clean_timestep = clean_timestep + self.fps_embedder(fps_tensor[:, None], batch)
            clean_modulation = self.t_block(clean_timestep)

        embedded = self.x_embedder(latent).reshape(batch, frames, spatial, -1)
        embedded = embedded + position[:, None]
        value = embedded.reshape(batch, frames * spatial, -1)
        control = None
        if self.control_depth:
            control_map = self._encode_control_map(
                static_maps, latent, frames, token_h, token_w
            )
            control = self.x_control_embedder(latent).reshape(
                batch, frames, spatial, -1
            )
            control = control + position[:, None] + self.before_proj(control_map)
            control = control.reshape(batch, frames * spatial, -1)

        for block_index, (spatial_block, temporal_block) in enumerate(
            zip(self.base_blocks_s, self.base_blocks_t)
        ):
            common = {
                "frames": frames,
                "spatial": spatial,
                "x_mask": x_mask,
                "clean_timestep_modulation": clean_modulation,
            }
            value = self._run_block(
                spatial_block,
                value,
                condition_tokens,
                timestep_modulation,
                **common,
            )
            if block_index < self.control_depth:
                control, control_skip = self._run_block(
                    self.control_blocks_s[block_index],
                    control,
                    condition_tokens,
                    timestep_modulation,
                    **common,
                )
                value = value + control_skip
            value = self._run_block(
                temporal_block,
                value,
                condition_tokens,
                timestep_modulation,
                rotary=self.rope.rotate_queries_or_keys,
                **common,
            )
            if block_index < self.control_depth:
                control, control_skip = self._run_block(
                    self.control_blocks_t[block_index],
                    control,
                    condition_tokens,
                    timestep_modulation,
                    rotary=self.rope.rotate_queries_or_keys,
                    **common,
                )
                value = value + control_skip
        value = self.final_layer(
            value,
            timestep,
            frames,
            spatial,
            x_mask=x_mask,
            clean_timestep=clean_timestep,
        )
        return self._unpatchify(
            value, frames, token_h, token_w, (real_t, real_h, real_w)
        ).float()
