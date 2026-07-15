from __future__ import annotations

from collections.abc import Mapping

import numpy as np


DEFAULT_CONTROL_GATE_THRESHOLDS = {
    # Video tensors use the model's [-1, 1] range.
    "min_counterfactual_video_mae": 0.005,
    # Motion uses Farneback pixels when OpenCV is installed and frame MAE otherwise.
    "max_stop_late_vs_straight": 0.85,
    "max_hold_late_vs_straight": 0.50,
    "max_stop_late_to_early": 1.00,
    # Positive image x is right. A left camera yaw should move static features right.
    "min_left_right_horizontal_separation": 0.05,
}


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _video_tchw(value) -> np.ndarray:
    value = _as_numpy(value)
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise ValueError("Control metrics currently require a single video per case")
        value = value[0]
    if value.ndim != 4 or value.shape[1] not in {1, 3}:
        raise ValueError(f"Expected [1,T,C,H,W] or [T,C,H,W], got {value.shape}")
    return value


def per_frame_pair_mae(video_a, video_b) -> np.ndarray:
    first, second = _video_tchw(video_a), _video_tchw(video_b)
    if first.shape != second.shape:
        raise ValueError("Counterfactual videos must have identical shapes")
    return np.abs(first - second).mean(axis=(1, 2, 3))


def pair_report(video_a, video_b) -> dict[str, object]:
    per_frame = per_frame_pair_mae(video_a, video_b)
    x = np.linspace(0.0, 1.0, len(per_frame), dtype=np.float64)
    slope = float(np.polyfit(x, per_frame.astype(np.float64), 1)[0]) if len(per_frame) > 1 else 0.0
    return {
        "video_mae": float(per_frame.mean()),
        "per_frame_video_mae": per_frame.tolist(),
        "first_video_mae": float(per_frame[0]),
        "last_video_mae": float(per_frame[-1]),
        "horizon_slope": slope,
    }


def _unit_rgb(frame: np.ndarray, *, signed_range: bool | None = None) -> np.ndarray:
    value = np.moveaxis(frame, 0, -1)
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    if signed_range is None:
        signed_range = float(value.min()) < -0.05
    if signed_range:
        value = (value + 1.0) * 0.5
    return np.clip(value, 0.0, 1.0)


def _gray_u8(frame: np.ndarray, *, signed_range: bool) -> np.ndarray:
    rgb = _unit_rgb(frame, signed_range=signed_range)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.round(gray * 255.0).astype(np.uint8)


def summarize_motion(values) -> dict[str, float | list[float]]:
    profile = np.asarray(values, dtype=np.float64)
    if profile.ndim != 1 or not len(profile):
        raise ValueError("Motion profile must be a non-empty vector")
    quarter = max(1, len(profile) // 4)
    early = float(profile[:quarter].mean())
    late = float(profile[-quarter:].mean())
    x = np.linspace(0.0, 1.0, len(profile), dtype=np.float64)
    slope = float(np.polyfit(x, profile, 1)[0]) if len(profile) > 1 else 0.0
    return {
        "per_frame": profile.tolist(),
        "mean": float(profile.mean()),
        "early_mean": early,
        "late_mean": late,
        "late_to_early": late / max(early, 1e-8),
        "horizon_slope": slope,
    }


def motion_report(video, anchor=None, backend: str = "auto") -> dict[str, object]:
    """Report apparent frame motion using fixed-seed generated RGB.

    Farneback flow is a lightweight diagnostic, not a replacement for calibrated
    visual odometry.  When OpenCV is unavailable the function falls back to frame
    MAE and leaves horizontal direction unavailable.
    """

    if backend not in {"auto", "farneback", "frame_mae"}:
        raise ValueError("backend must be auto, farneback, or frame_mae")
    frames = _video_tchw(video)
    if anchor is not None:
        anchor = _as_numpy(anchor)
        if anchor.ndim == 4 and anchor.shape[0] == 1:
            anchor = anchor[0]
        if anchor.ndim != 3 or anchor.shape != frames.shape[1:]:
            raise ValueError("anchor must be [C,H,W] and match the generated video")
        frames = np.concatenate([anchor[None], frames], axis=0)

    cv2 = None
    if backend != "frame_mae":
        try:
            import cv2 as cv2_module

            cv2 = cv2_module
        except ImportError:
            if backend == "farneback":
                raise RuntimeError("OpenCV is required for the Farneback motion backend")

    magnitudes: list[float] = []
    horizontal: list[float] | None = [] if cv2 is not None else None
    signed_range = float(frames.min()) < -0.05
    if cv2 is not None:
        gray = [_gray_u8(frame, signed_range=signed_range) for frame in frames]
        for previous, current in zip(gray, gray[1:]):
            flow = cv2.calcOpticalFlowFarneback(
                previous,
                current,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            border_y = max(1, flow.shape[0] // 20)
            border_x = max(1, flow.shape[1] // 20)
            center = flow[border_y:-border_y, border_x:-border_x]
            magnitude = np.linalg.norm(center, axis=-1)
            magnitudes.append(float(np.median(magnitude)))
            horizontal.append(float(np.median(center[..., 0])))
        selected_backend = "farneback"
    else:
        unit = [_unit_rgb(frame, signed_range=signed_range) for frame in frames]
        magnitudes = [
            float(np.abs(current - previous).mean())
            for previous, current in zip(unit, unit[1:])
        ]
        selected_backend = "frame_mae"

    report: dict[str, object] = {
        "backend": selected_backend,
        "magnitude": summarize_motion(magnitudes),
        "horizontal_flow": None,
    }
    if horizontal is not None:
        report["horizontal_flow"] = summarize_motion(horizontal)
    return report


def _check(value, operator: str, threshold: float) -> dict[str, object]:
    if operator == "<=":
        passed = value <= threshold
    elif operator == ">=":
        passed = value >= threshold
    else:
        raise ValueError(f"Unsupported gate operator: {operator}")
    return {
        "value": float(value),
        "operator": operator,
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def control_gate_report(
    motion: Mapping[str, Mapping[str, object]],
    pairs: Mapping[str, Mapping[str, object]],
    *,
    finite: Mapping[str, bool] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Evaluate pilot control gates without claiming a publication-grade metric."""

    limits = dict(DEFAULT_CONTROL_GATE_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(limits)
        if unknown:
            raise ValueError(f"Unknown control gate thresholds: {sorted(unknown)}")
        limits.update({key: float(value) for key, value in thresholds.items()})

    checks: dict[str, dict[str, object]] = {}
    unavailable: list[str] = []
    if finite is not None:
        value = all(finite.values())
        checks["all_outputs_finite"] = {
            "value": bool(value),
            "operator": "is",
            "threshold": True,
            "passed": bool(value),
        }

    for mode in ("stop", "hold", "shuffle", "invalid", "zero_kinematics"):
        pair = pairs.get(mode)
        if mode == "hold" and pair is None:
            pair = pairs.get("zero")
        if pair is None:
            unavailable.append(f"{mode}_counterfactual_effect")
            continue
        checks[f"{mode}_counterfactual_effect"] = _check(
            float(pair["video_mae"]),
            ">=",
            limits["min_counterfactual_video_mae"],
        )

    straight = motion.get("straight")
    stop = motion.get("stop")
    hold = motion.get("hold") or motion.get("zero")
    if straight is not None and stop is not None:
        straight_late = float(straight["magnitude"]["late_mean"])
        stop_late = float(stop["magnitude"]["late_mean"])
        checks["stop_late_motion_vs_straight"] = _check(
            stop_late / max(straight_late, 1e-8),
            "<=",
            limits["max_stop_late_vs_straight"],
        )
        checks["stop_motion_decreases"] = _check(
            float(stop["magnitude"]["late_to_early"]),
            "<=",
            limits["max_stop_late_to_early"],
        )
    else:
        unavailable.extend(["stop_late_motion_vs_straight", "stop_motion_decreases"])

    if straight is not None and hold is not None:
        straight_late = float(straight["magnitude"]["late_mean"])
        hold_late = float(hold["magnitude"]["late_mean"])
        checks["hold_late_motion_vs_straight"] = _check(
            hold_late / max(straight_late, 1e-8),
            "<=",
            limits["max_hold_late_vs_straight"],
        )
    else:
        unavailable.append("hold_late_motion_vs_straight")

    left = motion.get("left")
    right = motion.get("right")
    left_flow = left.get("horizontal_flow") if left is not None else None
    right_flow = right.get("horizontal_flow") if right is not None else None
    if left_flow is not None and right_flow is not None:
        separation = float(left_flow["mean"]) - float(right_flow["mean"])
        checks["left_right_direction"] = _check(
            separation,
            ">=",
            limits["min_left_right_horizontal_separation"],
        )
    else:
        unavailable.append("left_right_direction")

    failed = [name for name, check in checks.items() if not check["passed"]]
    status = "fail" if failed else ("incomplete" if unavailable else "pass")
    return {
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "unavailable_checks": unavailable,
        "thresholds": limits,
        "note": (
            "Pilot diagnostic only: pass/fail must be confirmed across fixed clips, seeds, "
            "step-zero/raw/EMA, and visual inspection."
        ),
    }
