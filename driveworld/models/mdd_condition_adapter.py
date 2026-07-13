from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = object
    F = None

from .magicdrive_single_view_stdit import (
    MagicMLP,
    MagicRotaryEmbedding,
    MagicSelfAttention,
    _modulate,
)


class MagicFourierEmbedder:
    """Exact ordering used by MagicDrive: input, then sin/cos for 1,2,4,8."""

    def __init__(self, input_dim: int = 3, num_freqs: int = 4, include_input: bool = True):
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.out_dim = input_dim * (int(include_input) + 2 * num_freqs)

    def __call__(self, value):
        outputs = [value] if self.include_input else []
        for exponent in range(self.num_freqs):
            frequency = float(2**exponent)
            outputs.extend([torch.sin(value * frequency), torch.cos(value * frequency)])
        return torch.cat(outputs, dim=-1)


def cog_temporal_downsample(value):
    """MagicDrive's cog_temp_down for [B,T,S,D], retaining the first odd frame."""
    if value.ndim != 4:
        raise ValueError("temporal downsample expects [B,T,S,D]")
    batch, frames, spatial, hidden = value.shape
    value = value.permute(0, 2, 3, 1).reshape(batch * spatial, hidden, frames)
    if frames % 2:
        first, rest = value[..., :1], value[..., 1:]
        if rest.shape[-1]:
            rest = F.avg_pool1d(rest, kernel_size=2, stride=2)
        value = torch.cat([first, rest], dim=-1)
    else:
        value = F.avg_pool1d(value, kernel_size=2, stride=2)
    return value.reshape(batch, spatial, hidden, -1).permute(0, 3, 1, 2)


class MagicCameraEmbedder(nn.Module if torch is not None else object):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        num: int,
        num_freqs: int = 4,
        after_proj: bool = True,
    ):
        super().__init__()
        self.embedder = MagicFourierEmbedder(input_dim, num_freqs)
        self.emb2token = nn.Linear(self.embedder.out_dim * num, out_dim)
        self.uncond_cam = nn.Parameter(torch.randn(input_dim, num))
        self.after_proj = nn.Linear(out_dim, out_dim) if after_proj else None

    def embed_cam(self, parameters, valid=None):
        if parameters.ndim != 3:
            raise ValueError("camera parameters must be [N,3|4,K]")
        if parameters.shape[1] == 4:
            parameters = parameters[:, :-1]
        if parameters.shape[1:] != self.uncond_cam.shape:
            raise ValueError(
                f"Expected camera parameter shape {tuple(self.uncond_cam.shape)}, "
                f"got {tuple(parameters.shape[1:])}"
            )
        if valid is not None:
            parameters = torch.where(
                valid.to(dtype=torch.bool)[:, None, None],
                parameters,
                self.uncond_cam[None],
            )
        batch, _, columns = parameters.shape
        values = parameters.permute(0, 2, 1).reshape(batch * columns, -1)
        embedded = self.embedder(values).reshape(batch, -1)
        token = self.emb2token(embedded)
        if self.after_proj is not None:
            token = self.after_proj(token)
        return token, embedded


class MagicFrameEmbedder(MagicCameraEmbedder):
    def __init__(
        self,
        out_dim: int = 1152,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__(input_dim=3, out_dim=out_dim, num=4, after_proj=False)
        self.rope = MagicRotaryEmbedding(out_dim // num_heads)
        self.norm1 = nn.LayerNorm(out_dim, eps=1e-6, elementwise_affine=False)
        self.attn = MagicSelfAttention(out_dim, num_heads, qkv_bias=True, qk_norm=True)
        self.scale_shift_table = nn.Parameter(torch.randn(6, out_dim) / out_dim**0.5)
        self.norm2 = nn.LayerNorm(out_dim, eps=1e-6, elementwise_affine=False)
        self.mlp = MagicMLP(out_dim, mlp_ratio)
        self.final_proj = nn.Linear(out_dim, out_dim)

    def embed_cam(self, parameters, valid=None, frames=None, spatial=1):
        if frames is None:
            raise ValueError("frames is required for temporal frame embedding")
        token, embedded = super().embed_cam(parameters, valid)
        if token.shape[0] % (frames * spatial):
            raise ValueError("flattened pose batch is incompatible with frames/spatial")
        batch = token.shape[0] // (frames * spatial)
        token = token.reshape(batch, frames, spatial, -1).permute(0, 2, 1, 3)
        token = token.reshape(batch * spatial, frames, -1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None]
        ).chunk(6, dim=1)
        attended = self.attn(
            _modulate(self.norm1(token), shift_attn, scale_attn),
            rotary=self.rope.rotate_queries_or_keys,
        )
        token = token + gate_attn * attended
        token = token + gate_mlp * self.mlp(
            _modulate(self.norm2(token), shift_mlp, scale_mlp)
        )
        token = token.reshape(batch, spatial, frames, -1).permute(0, 2, 1, 3)
        token = self.final_proj(token)
        token = cog_temporal_downsample(cog_temporal_downsample(token))
        return token, embedded


class KinematicsEmbedder(nn.Module if torch is not None else object):
    """New zero-residual adapter for velocity/acceleration/yaw-rate/steering."""

    def __init__(self, out_dim: int = 1152, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, kinematics, valid):
        if kinematics.shape != valid.shape or kinematics.shape[-1] != 6:
            raise ValueError("kinematics and valid must both be [B,T,6]")
        valid_float = valid.to(dtype=kinematics.dtype)
        values = torch.where(valid, kinematics, torch.zeros_like(kinematics))
        return self.mlp(torch.cat([values, valid_float], dim=-1))


class MagicNullBBoxEmbedder(nn.Module if torch is not None else object):
    """Stage-3 bbox temporal embedder restricted to the exact no-box token path.

    DriveWorld currently has no 3D detection annotations in its manifest.  MagicDrive
    represents that state with one learned null box token per frame; omitting the bbox
    branch altogether changes the cross-attention sequence seen by every Stage-3 block.
    All parameters are kept under the upstream ``bbox_embedder.*`` names so the EMA is
    loaded strictly, even though real box geometry/class paths remain intentionally
    disabled until the dataset exposes them.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        n_classes: int = 10,
        position_dim: int = 216,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mean_var = nn.Parameter(torch.randn(n_classes, 2))
        self.null_class_feature = nn.Parameter(torch.zeros(hidden_size))
        self.null_pos_feature = nn.Parameter(torch.zeros(position_dim))
        self.mask_class_feature = nn.Parameter(torch.zeros(hidden_size))
        self.mask_pos_feature = nn.Parameter(torch.zeros(position_dim))
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )
        self.register_buffer("_class_tokens", torch.randn(n_classes, hidden_size))
        self.bbox_proj = nn.Linear(position_dim, hidden_size)
        self.second_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, hidden_size),
        )
        self.rope = MagicRotaryEmbedding(hidden_size // num_heads)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = MagicSelfAttention(hidden_size, num_heads, qkv_bias=True, qk_norm=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = MagicMLP(hidden_size, mlp_ratio)
        self.final_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, batch_size: int, frames: int = 17):
        if frames != 17:
            raise ValueError("Stage-3 null bbox contract requires 17 RGB frames")
        position = self.bbox_proj(self.null_pos_feature)
        position = F.silu(position)
        token = self.second_linear(
            torch.cat([position, self.null_class_feature], dim=-1)
        )
        token = token.reshape(1, 1, -1).expand(batch_size, frames, -1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None]
        ).chunk(6, dim=1)
        attended = self.attn(
            _modulate(self.norm1(token), shift_attn, scale_attn),
            rotary=self.rope.rotate_queries_or_keys,
        )
        token = token + gate_attn * attended
        token = token + gate_mlp * self.mlp(
            _modulate(self.norm2(token), shift_mlp, scale_mlp)
        )
        token = self.final_proj(token)[:, :, None]
        return cog_temporal_downsample(cog_temporal_downsample(token))


def ego_to_next2top(ego):
    """Convert [x,y,yaw] in anchor coordinates to anchor->current 4x4 transforms.

    This matches MagicDrive obtain_next2top(v2=True): the matrix maps a point in
    the first/anchor frame into the coordinate system of the current frame.
    """
    if ego.ndim != 3 or ego.shape[-1] < 3:
        raise ValueError("ego must be [B,T,>=3]")
    x, y, yaw = ego[..., 0], ego[..., 1], ego[..., 2]
    cosine, sine = yaw.cos(), yaw.sin()
    transform = torch.zeros(*ego.shape[:2], 4, 4, device=ego.device, dtype=ego.dtype)
    transform[..., 0, 0] = cosine
    transform[..., 0, 1] = sine
    transform[..., 1, 0] = -sine
    transform[..., 1, 1] = cosine
    transform[..., 2, 2] = 1
    transform[..., 3, 3] = 1
    transform[..., 0, 3] = -(cosine * x + sine * y)
    transform[..., 1, 3] = sine * x - cosine * y
    return transform


class MDDConditionAdapter(nn.Module if torch is not None else object):
    """Stage-3 camera/frame conditioning plus a zero-init DriveWorld action residual."""

    def __init__(
        self,
        hidden_size: int = 1152,
        frame_num_heads: int = 8,
        kinematics_hidden_size: int = 256,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required")
        super().__init__()
        self.hidden_size = hidden_size
        self.camera_embedder = MagicCameraEmbedder(3, hidden_size, 7, after_proj=True)
        self.frame_embedder = MagicFrameEmbedder(hidden_size, frame_num_heads)
        self.bbox_embedder = MagicNullBBoxEmbedder(
            hidden_size, frame_num_heads
        )
        self.kinematics_embedder = KinematicsEmbedder(
            hidden_size, kinematics_hidden_size
        )

    def forward(
        self,
        ego,
        ego_valid,
        *,
        base_token,
        camera_parameters=None,
        camera_valid=None,
    ):
        if ego.ndim != 3 or ego.shape[-1] != 9:
            raise ValueError("ego must be [B,17,9]")
        if ego.shape[1] != 17 or ego_valid.shape != ego.shape:
            raise ValueError("Stage-3 I2V condition requires ego/valid [B,17,9]")
        batch, frames, _ = ego.shape
        pose_valid = ego_valid[..., :3].all(dim=-1)
        poses = ego_to_next2top(ego)
        frame_token, _ = self.frame_embedder.embed_cam(
            poses.reshape(batch * frames, 4, 4),
            pose_valid.reshape(-1),
            frames=frames,
            spatial=1,
        )
        kinematics = self.kinematics_embedder(ego[..., 3:], ego_valid[..., 3:])[:, :, None]
        kinematics = cog_temporal_downsample(cog_temporal_downsample(kinematics))
        frame_token = frame_token + kinematics

        if camera_parameters is None:
            camera_parameters = self.camera_embedder.uncond_cam[None].expand(batch, -1, -1)
            camera_valid = torch.zeros(batch, device=ego.device, dtype=torch.bool)
        elif camera_valid is None:
            camera_valid = torch.ones(batch, device=ego.device, dtype=torch.bool)
        camera_token, _ = self.camera_embedder.embed_cam(camera_parameters, camera_valid)
        latent_frames = frame_token.shape[1]
        camera_token = camera_token[:, None, None].expand(-1, latent_frames, -1, -1)
        base = base_token.to(dtype=frame_token.dtype)[None, None, None]
        text_token = base.expand(batch, latent_frames, 1, -1)
        bbox_token = self.bbox_embedder(batch, frames) + base
        return torch.cat(
            [frame_token + base, camera_token + base, text_token, bbox_token], dim=2
        )
