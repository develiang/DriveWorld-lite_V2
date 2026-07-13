from __future__ import annotations


def charbonnier_loss(prediction, target, epsilon: float = 1e-3):
    return ((prediction - target).square() + epsilon**2).sqrt().mean()


def temporal_difference_loss(prediction, target):
    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    return (pred_delta - target_delta).abs().mean()

