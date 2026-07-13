from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from driveworld.evaluation.horizon_metrics import horizon_report, per_frame_edge_energy


def test_horizon_metrics_report_per_frame_degradation():
    target = torch.zeros(1, 3, 3, 8, 8)
    target[:, :, :, :, 4:] = 1
    prediction = target.clone()
    prediction[:, 2] = 0.5
    report = horizon_report(prediction, target)
    assert report["psnr"].shape == (3,)
    assert report["mae"][2] > report["mae"][0]
    assert report["edge_retention"][2] < report["edge_retention"][0]
    assert per_frame_edge_energy(target).shape == (3,)

