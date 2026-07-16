from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    """Linear layer retaining upstream weight/bias names plus a zero-init LoRA delta."""

    def __init__(self, *args, rank: int, alpha: float, dropout: float, **kwargs):
        super().__init__(*args, **kwargs)
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=self.weight.device, dtype=self.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, rank: int, alpha: float, dropout: float):
        converted = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        converted.weight = linear.weight
        converted.bias = linear.bias
        converted.weight.requires_grad_(False)
        if converted.bias is not None:
            converted.bias.requires_grad_(False)
        return converted

    def forward(self, value):
        base = F.linear(value, self.weight, self.bias)
        delta = F.linear(F.linear(self.lora_dropout(value), self.lora_A), self.lora_B)
        return base + delta * self.scaling


def inject_mdd_lora(
    denoiser: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    temporal: bool = True,
    cross_attention: bool = True,
    spatial_attention: bool = False,
):
    """Inject LoRA into selected temporal, cross-, and spatial-attention linears."""
    selected = []
    for name, module in list(denoiser.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        is_temporal = name.startswith(("base_blocks_t.", "control_blocks_t."))
        is_cross_attention = ".cross_attn." in name
        is_spatial_attention = name.startswith(
            ("base_blocks_s.", "control_blocks_s.")
        ) and ".attn." in name
        if not (
            (temporal and is_temporal)
            or (cross_attention and is_cross_attention)
            or (spatial_attention and is_spatial_attention)
        ):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = denoiser.get_submodule(parent_name)
        setattr(
            parent,
            child_name,
            LoRALinear.from_linear(
                module, rank=rank, alpha=alpha, dropout=dropout
            ),
        )
        selected.append(name)
    if not selected:
        raise RuntimeError("LoRA target policy matched no Stage-3 linear layers")
    return selected
