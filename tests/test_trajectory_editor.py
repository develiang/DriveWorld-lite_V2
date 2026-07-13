import numpy as np

from driveworld.control import edit_trajectory


def _trajectory():
    value = np.zeros((16, 9), dtype=np.float32)
    value[:, 0] = np.arange(1, 17)
    value[:, 3] = 6
    return value


def test_turn_sign_and_stop():
    left = edit_trajectory(_trajectory(), "left")
    right = edit_trajectory(_trajectory(), "right")
    stop = edit_trajectory(_trajectory(), "stop")
    assert left[-1, 1] > 0
    assert right[-1, 1] < 0
    assert left[-1, 2] > 0 > right[-1, 2]
    assert np.linalg.norm(stop[-1, 3:5]) == 0
    assert np.isfinite(left).all()

