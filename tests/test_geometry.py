import numpy as np

from driveworld.data.geometry import quaternion_to_yaw, relative_ego_features, wrap_angle


def test_quaternion_yaw_and_wrap():
    assert np.isclose(quaternion_to_yaw([1, 0, 0, 0]), 0)
    assert np.isclose(quaternion_to_yaw([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]), np.pi / 2)
    assert np.isclose(wrap_angle(3 * np.pi), -np.pi)


def test_relative_features_straight_motion():
    timestamps = np.arange(24, dtype=np.int64) * 1_000_000
    positions = np.column_stack([np.arange(24), np.zeros(24)])
    features = relative_ego_features(timestamps, positions, np.zeros(24), np.zeros(24), 7)
    assert np.allclose(features[7, :3], 0)
    assert np.allclose(features[:, 3], 1)
    assert np.allclose(features[:, 4:8], 0)

