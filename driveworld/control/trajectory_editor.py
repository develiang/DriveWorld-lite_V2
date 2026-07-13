from __future__ import annotations

import numpy as np


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def _reconstruct(speed: np.ndarray, yaw: np.ndarray, steering: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.column_stack([speed * np.cos(yaw), speed * np.sin(yaw)])
    position = np.cumsum(velocity * dt, axis=0)
    position -= velocity[0] * dt  # first future position begins one dt after anchor.
    position += velocity[0] * dt
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    yaw_rate = np.gradient(yaw, dt, edge_order=2)
    return np.column_stack([position, yaw, velocity, acceleration, yaw_rate, steering]).astype(
        np.float32
    )


def edit_trajectory(
    future_ego: np.ndarray,
    mode: str,
    fps: float = 6.0,
    turn_yaw_degrees: float = 25.0,
) -> np.ndarray:
    """Create a continuous, reproducible counterfactual future Ego trajectory."""
    ego = np.asarray(future_ego, dtype=np.float64)
    if ego.ndim != 2 or ego.shape[1] != 9:
        raise ValueError("future_ego must have shape [T,9]")
    dt = 1.0 / fps
    progress = np.arange(1, len(ego) + 1) / len(ego)
    profile = _smoothstep(progress)
    speed = np.linalg.norm(ego[:, 3:5], axis=1)
    # Preserve a plausible initial speed even if derivative noise creates a short zero.
    initial_speed = float(np.median(speed[: min(3, len(speed))]))
    speed = np.maximum(speed, 0.0)
    steering = ego[:, 8].copy()

    if mode == "straight":
        yaw = ego[:, 2] * (1 - profile)
        steering *= 1 - profile
    elif mode in {"left", "right"}:
        sign = 1.0 if mode == "left" else -1.0
        target_yaw = np.deg2rad(turn_yaw_degrees) * sign
        yaw = target_yaw * profile
        # Steering wheel condition follows yaw-rate sign; magnitude is deliberately conservative.
        steering = sign * np.maximum(np.abs(steering), 0.15) * np.sin(np.pi * progress)
    elif mode == "stop":
        yaw = ego[:, 2]
        speed = initial_speed * (1 - profile)
        speed[-1] = 0.0
    elif mode == "original":
        return ego.astype(np.float32, copy=True)
    else:
        raise ValueError(f"Unknown trajectory mode: {mode}")
    return _reconstruct(speed, yaw, steering, dt)

