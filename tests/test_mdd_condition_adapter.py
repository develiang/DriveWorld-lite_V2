from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from driveworld.models.mdd_condition_adapter import (  # noqa: E402
    MagicNullBBoxEmbedder,
    MDDConditionAdapter,
    cog_temporal_downsample,
    ego_to_next2top,
)


def test_ego_to_next2top_has_magicdrive_v2_direction():
    ego = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, math.pi / 2]]])
    transform = ego_to_next2top(ego)
    assert torch.allclose(transform[0, 0, :2, 3], torch.tensor([-1.0, 0.0]))
    assert torch.allclose(
        transform[0, 1, :2, :2], torch.tensor([[0.0, 1.0], [-1.0, 0.0]]), atol=1e-6
    )
    assert torch.allclose(transform[0, 1, :2, 3], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_cog_temporal_downsample_maps_17_to_5_and_keeps_first():
    value = torch.arange(17, dtype=torch.float32).reshape(1, 17, 1, 1)
    once = cog_temporal_downsample(value)
    twice = cog_temporal_downsample(once)
    assert once.shape[1] == 9
    assert twice.shape[1] == 5
    assert twice[0, 0, 0, 0] == 0


def test_cog_temporal_downsample_maps_24_to_6():
    value = torch.arange(24, dtype=torch.float32).reshape(1, 24, 1, 1)
    assert cog_temporal_downsample(cog_temporal_downsample(value)).shape[1] == 6


def test_condition_adapter_stage3_shapes_and_zero_kinematics_residual():
    torch.manual_seed(5)
    adapter = MDDConditionAdapter(
        hidden_size=32, frame_num_heads=4, kinematics_hidden_size=16
    ).eval()
    ego = torch.randn(2, 17, 9)
    ego[:, 0, :3] = 0
    valid = torch.ones_like(ego, dtype=torch.bool)
    base_token = torch.randn(32)
    output = adapter(ego, valid, base_token=base_token)
    changed = ego.clone()
    changed[..., 3:] += 100
    changed_output = adapter(changed, valid, base_token=base_token)
    assert output.shape == (2, 5, 4, 32)
    assert torch.allclose(output, changed_output)


def test_null_bbox_token_has_exact_stage3_temporal_contract():
    embedder = MagicNullBBoxEmbedder(hidden_size=32, num_heads=4).eval()
    output = embedder(2)
    assert output.shape == (2, 5, 1, 32)
    assert torch.isfinite(output).all()


def test_condition_adapter_supports_24_rgb_frames():
    adapter = MDDConditionAdapter(
        hidden_size=32, frame_num_heads=4, kinematics_hidden_size=16
    ).eval()
    ego = torch.randn(1, 24, 9)
    output = adapter(
        ego,
        torch.ones_like(ego, dtype=torch.bool),
        base_token=torch.randn(32),
    )
    assert output.shape == (1, 6, 4, 32)
    assert adapter.bbox_embedder(1, frames=24).shape == (1, 6, 1, 32)


def test_condition_adapter_parameter_names_match_stage3():
    with torch.device("meta"):
        adapter = MDDConditionAdapter()
    state = adapter.state_dict()
    assert state["camera_embedder.emb2token.weight"].shape == (1152, 189)
    assert state["frame_embedder.emb2token.weight"].shape == (1152, 108)
    assert state["frame_embedder.rope.freqs"].shape == (72,)
    assert state["frame_embedder.attn.q_norm.weight"].shape == (144,)
    assert state["frame_embedder.final_proj.weight"].shape == (1152, 1152)
    assert state["bbox_embedder.null_pos_feature"].shape == (216,)
    assert state["bbox_embedder.second_linear.0.weight"].shape == (512, 2304)
    assert state["bbox_embedder.rope.freqs"].shape == (72,)
    assert state["bbox_embedder.final_proj.weight"].shape == (1152, 1152)
