import numpy as np
import pytest
import xarray as xr

from movement.kinematics.trajectory_complexity import compute_turning_angles


def make_trajectory(positions: list[list[float]]) -> xr.DataArray:
    """Create a 2D trajectory DataArray for testing."""
    return xr.DataArray(
        positions,
        dims=["time", "space"],
        coords={"time": np.arange(len(positions)), "space": ["x", "y"]},
    )


def test_turning_angles_straight_line():
    """A straight path should yield 0.0 turning angles."""
    pos = make_trajectory([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    angles = compute_turning_angles(pos)

    # The first two frames cannot form an angle, so they naturally become NaN
    assert np.isnan(angles[0:2]).all()
    # The remaining frames should be exactly 0 radians
    assert np.isclose(angles[2:], 0.0).all()


def test_turning_angles_right_turn():
    """Moving +x then +y should result in a 90-degree (pi/2) left turn."""
    pos = make_trajectory([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    angles = compute_turning_angles(pos)

    assert np.isnan(angles[0:2]).all()
    assert np.isclose(angles[2], np.pi / 2)


def test_turning_angles_stationary_animal():
    """Zero-length steps should be masked as
    NaN to prevent arctan2(0,0) artifacts.
    """
    # The animal moves, STOPS at [1,1] for a frame, then moves again
    pos = make_trajectory([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    angles = compute_turning_angles(pos)

    # Because step 2 has zero length, any turn involving step 2 must be NaN
    assert np.isnan(angles[2])
    assert np.isnan(angles[3])


def test_turning_angles_rejects_3d():
    """The function should raise an error if given 3D spatial data."""
    pos = xr.DataArray(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dims=["time", "space"],
        coords={"time": [0, 1], "space": ["x", "y", "z"]},
    )

    with pytest.raises(ValueError, match="only support 2D"):
        compute_turning_angles(pos)
