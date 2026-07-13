from __future__ import annotations

import numpy as np


def frame_difference_error(prediction, target) -> float:
    if hasattr(prediction, "detach"):
        pred_delta = prediction[:, 1:] - prediction[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        return (pred_delta - target_delta).abs().mean().item()
    prediction, target = np.asarray(prediction), np.asarray(target)
    return float(np.mean(np.abs(np.diff(prediction, axis=1) - np.diff(target, axis=1))))


def condition_sensitivity(video_a, video_b, trajectory_a, trajectory_b, epsilon=1e-6) -> float:
    if hasattr(video_a, "detach"):
        video_distance = (video_a - video_b).abs().mean().item()
        trajectory_distance = (trajectory_a - trajectory_b).abs().mean().item()
    else:
        video_distance = float(np.mean(np.abs(np.asarray(video_a) - np.asarray(video_b))))
        trajectory_distance = float(
            np.mean(np.abs(np.asarray(trajectory_a) - np.asarray(trajectory_b)))
        )
    return video_distance / max(trajectory_distance, epsilon)

