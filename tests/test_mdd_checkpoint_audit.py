from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from scripts.audit_mdd_stage3_checkpoint import (  # noqa: E402
    audit_checkpoint,
    classify_key,
    summarize_mapping,
    summarize_state_dict,
)


def test_classify_magicdrive_parameter_groups():
    assert classify_key("base_blocks_s.0.attn.qkv.weight") == "base_spatial"
    assert classify_key("base_blocks_t.0.attn.qkv.weight") == "base_temporal"
    assert classify_key("base_blocks_s.0.cross_view_attn.qkv.weight") == "cross_view"
    assert classify_key("control_blocks_s.0.after_proj.weight") == "control_spatial"
    assert classify_key("frame_embedder.final_proj.weight") == "condition_embedder"


def test_summarize_state_dict_infers_stage3_shape():
    state = {
        "x_embedder.proj.weight": torch.zeros(12, 16, 1, 2, 2),
        "base_blocks_s.0.attn.qkv.weight": torch.zeros(36, 12),
        "base_blocks_t.0.attn.qkv.weight": torch.zeros(36, 12),
        "final_layer.linear.weight": torch.zeros(64, 12),
    }
    report = summarize_state_dict(state)
    assert report["tensor_count"] == 4
    assert report["architecture"]["hidden_size"] == 12
    assert report["architecture"]["in_channels"] == 16
    assert report["architecture"]["patch_size"] == [1, 2, 2]
    assert report["architecture"]["base_blocks_s_depth"] == 1


def test_audit_checkpoint_uses_plain_ema_state_dict(tmp_path):
    checkpoint = tmp_path / "ema.pt"
    torch.save({"x_embedder.proj.weight": torch.zeros(8, 16, 1, 2, 2)}, checkpoint)
    report = audit_checkpoint(checkpoint, include_sha256=True)
    assert report["format"] == "mdd-stage3-ema-audit-v1"
    assert report["tensor_count"] == 1
    assert len(report["sha256"]) == 64


def test_mapping_report_uses_both_target_and_source_denominators():
    source = {
        "x_embedder.proj.weight": torch.zeros(8, 4, 1, 2, 2),
        "base_blocks_s.0.cross_view_attn.qkv.weight": torch.zeros(24, 8),
    }
    target = {
        "x_embedder.proj.weight": torch.empty(8, 4, 1, 2, 2, device="meta")
    }
    report = summarize_mapping(source, target)
    assert report["target_coverage"] == 1.0
    assert report["source_coverage"] < 1.0
    assert report["unused_groups"]["cross_view"]["tensors"] == 1
