import numpy as np

from driveworld.data.can_interpolator import _interp_columns


def test_interpolation_and_validity():
    values, valid = _interp_columns(
        np.array([0, 10]), np.array([[0.0], [20.0]]), np.array([-1, 0, 5, 10, 11])
    )
    assert np.allclose(values[:, 0], [0, 0, 10, 20, 20])
    assert valid.tolist() == [False, True, True, True, False]

