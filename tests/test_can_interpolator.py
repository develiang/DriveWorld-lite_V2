import json

import numpy as np

import driveworld.data.can_interpolator as can_module
from driveworld.data.can_interpolator import CanBusInterpolator, _interp_columns


def test_interpolation_and_validity():
    values, valid = _interp_columns(
        np.array([0, 10]), np.array([[0.0], [20.0]]), np.array([-1, 0, 5, 10, 11])
    )
    assert np.allclose(values[:, 0], [0, 0, 10, 20, 20])
    assert valid.tolist() == [False, True, True, True, False]


def test_scene_messages_are_loaded_and_prepared_once(tmp_path, monkeypatch):
    pose = [
        {"utime": 0, "pos": [0, 0, 0], "orientation": [1, 0, 0, 0]},
        {"utime": 10, "pos": [20, 0, 0], "orientation": [1, 0, 0, 0]},
    ]
    steering = [{"utime": 0, "value": 0.0}, {"utime": 10, "value": 1.0}]
    (tmp_path / "scene-0001_pose.json").write_text(json.dumps(pose), encoding="utf-8")
    (tmp_path / "scene-0001_steeranglefeedback.json").write_text(
        json.dumps(steering), encoding="utf-8"
    )
    original_load = can_module._load_messages
    loads = []

    def counted_load(path):
        loads.append(path)
        return original_load(path)

    monkeypatch.setattr(can_module, "_load_messages", counted_load)
    interpolator = CanBusInterpolator(tmp_path)
    fallback_t = np.array([0, 10])
    fallback_position = np.zeros((2, 2))
    fallback_yaw = np.zeros(2)

    prepared = interpolator.prepare_scene(
        "scene-0001", fallback_t, fallback_position, fallback_yaw
    )
    assert (
        interpolator.prepare_scene(
            "scene-0001", fallback_t, fallback_position, fallback_yaw
        )
        is prepared
    )
    value = interpolator.interpolate_prepared(prepared, np.array([5]))

    assert len(loads) == 2
    assert np.allclose(value.positions_world, [[10, 0]])
    assert np.allclose(value.steering, [0.5])
    assert value.pose_valid.tolist() == [True]
    assert value.steering_valid.tolist() == [True]
