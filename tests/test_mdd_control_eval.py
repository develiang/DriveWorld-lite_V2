from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from scripts.generate_mdd_counterfactual_demo import (  # noqa: E402
    _condition_variant,
    _evaluate_case,
)


def _item(clip_id: str, offset: float = 0.0):
    trajectory = torch.zeros(16, 9)
    trajectory[:, 0] = torch.arange(1, 17) + offset
    trajectory[:, 3] = 6 + offset
    return {
        "clip_id": clip_id,
        "future_ego_raw": trajectory,
        "future_ego_valid": torch.ones_like(trajectory, dtype=torch.bool),
    }


def test_control_variants_cover_hold_invalid_shuffle_and_branch_ablation():
    item, shuffled = _item("source"), _item("shuffle", offset=2)

    hold, hold_valid, hold_source = _condition_variant(item, shuffled, "hold", 6.0, 25.0)
    assert np.count_nonzero(hold) == 0
    assert hold_valid.all()
    assert hold_source["kind"] == "zero_ego_hold"

    invalid, invalid_valid, invalid_source = _condition_variant(
        item, shuffled, "invalid", 6.0, 25.0
    )
    assert np.array_equal(invalid, item["future_ego_raw"].numpy())
    assert not invalid_valid.any()
    assert invalid_source["kind"] == "future_ego_invalid"

    other, other_valid, other_source = _condition_variant(
        item, shuffled, "shuffle", 6.0, 25.0
    )
    assert np.array_equal(other, shuffled["future_ego_raw"].numpy())
    assert other_valid.all()
    assert other_source["clip_id"] == "shuffle"

    zero_kinematics, _, source = _condition_variant(
        item, shuffled, "zero_kinematics", 6.0, 25.0
    )
    assert np.array_equal(zero_kinematics[:, :3], item["future_ego_raw"].numpy()[:, :3])
    assert np.count_nonzero(zero_kinematics[:, 3:]) == 0
    assert source["kind"] == "zero_kinematics_pose_preserved"


class _FakeControlModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))

    def sample(self, past_rgb, future_ego, future_valid, **kwargs):
        del kwargs
        values = torch.where(future_valid, future_ego, torch.zeros_like(future_ego))
        signal = values.sum(dim=-1) / 100
        batch, frames = signal.shape
        height, width = past_rgb.shape[-2:]
        return signal.reshape(batch, frames, 1, 1, 1).expand(
            batch, frames, 3, height, width
        )


def _full_item(clip_id: str, offset: float = 0.0):
    item = _item(clip_id, offset)
    item.update(
        {
            "past_rgb": torch.zeros(1, 3, 8, 8),
            "future_rgb": torch.zeros(16, 3, 8, 8),
            "past_ego_raw": torch.zeros(1, 9),
            "past_ego_valid": torch.ones(1, 9, dtype=torch.bool),
        }
    )
    return item


def test_control_evaluation_case_writes_machine_readable_report(tmp_path):
    modes = (
        "original",
        "straight",
        "left",
        "right",
        "stop",
        "hold",
        "shuffle",
        "invalid",
        "zero_kinematics",
    )
    report = _evaluate_case(
        model=_FakeControlModel(),
        item=_full_item("source"),
        shuffle_item=_full_item("shuffle", offset=2),
        modes=modes,
        seed=42,
        num_steps=2,
        guidance=1.0,
        turn_yaw_degrees=25.0,
        fps=6.0,
        output_dir=tmp_path,
        weights="fake",
        checkpoint=None,
        checkpoint_step=None,
        motion_backend="frame_mae",
        gate_thresholds={},
        save_gifs=False,
    )
    assert report["format"] == "mdd-control-eval-v1"
    assert report["condition_sources"]["shuffle"]["clip_id"] == "shuffle"
    assert report["motion_backend"] == "frame_mae"
    assert report["grid"] is None
    assert (tmp_path / "metadata.json").is_file()
