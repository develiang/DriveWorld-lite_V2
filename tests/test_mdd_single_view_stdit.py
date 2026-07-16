from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from driveworld.models.magicdrive_single_view_stdit import (  # noqa: E402
    MagicDriveSingleViewSTDiT,
)


def test_mdd_single_view_tiny_forward_and_mask_contract():
    model = MagicDriveSingleViewSTDiT(
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        patch_size=(1, 2, 2),
    )
    latent = torch.randn(1, 4, 5, 8, 8)
    condition = torch.randn(1, 5, 3, 32)
    mask = torch.tensor([[False, True, True, True, True]])
    output = model(
        latent,
        torch.tensor([500.0]),
        condition,
        fps=12.0,
        height=64,
        width=64,
        x_mask=mask,
    )
    assert output.shape == latent.shape
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_mdd_single_view_has_stage3_base_parameter_names_without_cross_view():
    model = MagicDriveSingleViewSTDiT(
        in_channels=4, hidden_size=32, depth=1, num_heads=4
    )
    state = model.state_dict()
    assert state["base_blocks_s.0.attn.qkv.weight"].shape == (96, 32)
    assert state["base_blocks_t.0.cross_attn.kv_linear.weight"].shape == (64, 32)
    assert state["final_layer.scale_shift_table"].shape == (2, 32)
    assert not any("cross_view_attn" in name or "mva_proj" in name for name in state)


def test_full_stage3_single_view_can_be_described_on_meta_device():
    with torch.device("meta"):
        model = MagicDriveSingleViewSTDiT()
    state = model.state_dict()
    assert state["x_embedder.proj.weight"].shape == (1152, 16, 1, 2, 2)
    assert len(model.base_blocks_s) == 28
    assert len(model.base_blocks_t) == 28
    assert state["final_layer.linear.weight"].shape == (64, 1152)


def test_mdd_single_view_control_branch_has_stage3_names_and_runs_zero_map():
    model = MagicDriveSingleViewSTDiT(
        in_channels=4,
        hidden_size=32,
        depth=2,
        control_depth=1,
        num_heads=4,
        mlp_ratio=2,
        zero_map_size=16,
    )
    state = model.state_dict()
    assert state["x_control_embedder.proj.weight"].shape == (32, 4, 1, 2, 2)
    assert state["control_blocks_s.0.after_proj.weight"].shape == (32, 32)
    assert state["control_blocks_t.0.after_proj.weight"].shape == (32, 32)
    assert state["controlnet_cond_embedder.conv_in.weight"].shape == (16, 8, 3, 3)
    assert state["controlnet_cond_embedder_temp.conv_blocks.1.conv.weight"].shape == (
        16,
        16,
        3,
        3,
    )
    latent = torch.randn(1, 4, 5, 8, 8)
    output = model(
        latent,
        torch.tensor([500.0]),
        torch.randn(1, 5, 3, 32),
        fps=12.0,
        height=64,
        width=64,
        x_mask=torch.tensor([[False, True, True, True, True]]),
    )
    assert output.shape == latent.shape
    assert torch.isfinite(output).all()

    temporal_output = model(
        torch.randn(1, 4, 6, 8, 8),
        torch.tensor([500.0]),
        torch.randn(1, 6, 3, 32),
        fps=12.0,
        height=64,
        width=64,
        x_mask=torch.tensor([[False, False, True, True, True, True]]),
        rgb_frames=24,
    )
    assert temporal_output.shape == (1, 4, 6, 8, 8)
    assert torch.isfinite(temporal_output).all()
