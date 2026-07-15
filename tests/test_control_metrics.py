from __future__ import annotations

import numpy as np

from driveworld.evaluation.control_metrics import (
    control_gate_report,
    motion_report,
    pair_report,
    summarize_motion,
)


def _video(value: float = 0.0):
    return np.full((1, 4, 3, 8, 8), value, dtype=np.float32)


def test_pair_report_tracks_per_frame_counterfactual_growth():
    reference = _video()
    changed = reference.copy()
    changed[:, 1] = 0.1
    changed[:, 2] = 0.2
    changed[:, 3] = 0.3
    report = pair_report(changed, reference)
    assert np.allclose(report["per_frame_video_mae"], [0.0, 0.1, 0.2, 0.3])
    assert np.isclose(report["video_mae"], 0.15)
    assert report["horizon_slope"] > 0


def test_frame_mae_motion_backend_reports_decreasing_motion():
    video = _video()
    video[:, 0] = 0.8
    video[:, 1] = 1.0
    video[:, 2] = 1.0
    video[:, 3] = 1.0
    anchor = np.zeros((3, 8, 8), dtype=np.float32)
    report = motion_report(video, anchor=anchor, backend="frame_mae")
    assert report["backend"] == "frame_mae"
    assert report["horizontal_flow"] is None
    assert report["magnitude"]["early_mean"] > report["magnitude"]["late_mean"]


def test_control_gate_passes_directional_and_speed_relations():
    def motion(early, late, horizontal):
        magnitude = summarize_motion([early, early, late, late])
        flow = summarize_motion([horizontal] * 4)
        return {"backend": "farneback", "magnitude": magnitude, "horizontal_flow": flow}

    motions = {
        "straight": motion(2.0, 2.0, 0.0),
        "stop": motion(1.5, 0.8, 0.0),
        "hold": motion(0.4, 0.3, 0.0),
        "left": motion(2.0, 2.0, 0.2),
        "right": motion(2.0, 2.0, -0.2),
    }
    pairs = {
        mode: {"video_mae": 0.02}
        for mode in ("stop", "hold", "shuffle", "invalid", "zero_kinematics")
    }
    report = control_gate_report(
        motions,
        pairs,
        finite={mode: True for mode in motions},
    )
    assert report["status"] == "pass"
    assert not report["failed_checks"]


def test_control_gate_is_incomplete_when_direction_backend_is_unavailable():
    profiles = {
        "straight": summarize_motion([2.0, 2.0, 2.0, 2.0]),
        "stop": summarize_motion([1.5, 1.5, 0.8, 0.8]),
        "hold": summarize_motion([0.4, 0.4, 0.3, 0.3]),
        "left": summarize_motion([2.0, 2.0, 2.0, 2.0]),
        "right": summarize_motion([2.0, 2.0, 2.0, 2.0]),
    }
    motions = {
        mode: {"backend": "frame_mae", "magnitude": profile, "horizontal_flow": None}
        for mode, profile in profiles.items()
    }
    pairs = {
        mode: {"video_mae": 0.02}
        for mode in ("stop", "hold", "shuffle", "invalid", "zero_kinematics")
    }
    report = control_gate_report(motions, pairs)
    assert report["status"] == "incomplete"
    assert "left_right_direction" in report["unavailable_checks"]
