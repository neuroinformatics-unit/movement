"""Test that kinematics functions deal with 3D data, or refuse it clearly.

Functions that are defined for any number of spatial dimensions must give the
right answer for ``space = [x, y, z]``, and functions that are specific to 2D
must raise a ``ValueError`` instead of silently using only some of the
coordinates.
"""

import numpy as np
import pytest
import xarray as xr

from movement import kinematics
from movement.utils import vector

N_FRAMES = 21
TIME = np.linspace(0, 2, N_FRAMES)
DIRECTION = np.array([1.0, 2.0, 2.0])  # length 3


def _position(values: np.ndarray, space: list[str]) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=["time", "space"],
        coords={"time": TIME, "space": space},
    )


@pytest.fixture
def straight_line_3d() -> xr.DataArray:
    """Return a track with constant velocity (1, 2, 2), so speed 3."""
    return _position(TIME[:, None] * DIRECTION, ["x", "y", "z"])


@pytest.fixture
def helix_3d() -> xr.DataArray:
    """Return a helix of radius 1 that rises by 0.5 per time unit."""
    values = np.stack([np.cos(2 * TIME), np.sin(2 * TIME), 0.5 * TIME], axis=1)
    return _position(values, ["x", "y", "z"])


def test_velocity_speed_and_acceleration_3d(straight_line_3d):
    """Derivatives and the speed use all three coordinates."""
    velocity = kinematics.compute_velocity(straight_line_3d)
    speed = kinematics.compute_speed(straight_line_3d)
    acceleration = kinematics.compute_acceleration(straight_line_3d)

    assert velocity.sizes["space"] == 3
    np.testing.assert_allclose(
        velocity.values, np.tile(DIRECTION, (N_FRAMES, 1))
    )
    np.testing.assert_allclose(speed.values, 3.0)
    np.testing.assert_allclose(acceleration.values, 0.0, atol=1e-9)


def test_path_length_and_straightness_3d(helix_3d, straight_line_3d):
    """Path length sums the 3D step lengths; a line is perfectly straight."""
    steps = np.linalg.norm(np.diff(helix_3d.values, axis=0), axis=1)
    chord = np.linalg.norm(helix_3d.values[-1] - helix_3d.values[0])

    length = float(kinematics.compute_path_length(helix_3d))
    straightness = float(kinematics.compute_path_straightness(helix_3d))
    assert np.isclose(length, steps.sum())
    assert np.isclose(straightness, chord / steps.sum())

    line_length = float(kinematics.compute_path_length(straight_line_3d))
    line_straightness = float(
        kinematics.compute_path_straightness(straight_line_3d)
    )
    assert np.isclose(line_length, 6.0)
    assert np.isclose(line_straightness, 1.0)


def test_displacements_3d(straight_line_3d):
    """Forward and backward displacement keep the z component."""
    forward = kinematics.compute_forward_displacement(straight_line_3d)
    backward = kinematics.compute_backward_displacement(straight_line_3d)
    step = np.tile(DIRECTION * (TIME[1] - TIME[0]), (N_FRAMES - 1, 1))

    assert forward.sizes["space"] == backward.sizes["space"] == 3
    np.testing.assert_allclose(forward.values[:-1], step)
    # the backward displacement points from the current to the previous frame
    np.testing.assert_allclose(backward.values[1:], -step)


def test_norm_and_unit_vector_3d(helix_3d):
    """The norm and unit vectors are taken over x, y and z."""
    norm = vector.compute_norm(helix_3d)
    unit = vector.convert_to_unit(helix_3d)

    np.testing.assert_allclose(
        norm.values, np.linalg.norm(helix_3d.values, axis=1)
    )
    np.testing.assert_allclose(vector.compute_norm(unit).values, 1.0)


@pytest.mark.parametrize(
    "function",
    [
        kinematics.compute_turning_angle,
        kinematics.compute_directional_change,
        kinematics.compute_path_sinuosity,
        kinematics.compute_path_emax,
    ],
)
def test_planar_path_metrics_refuse_3d(function, helix_3d):
    """Metrics that are defined on the plane must not accept a z coordinate."""
    with pytest.raises(ValueError, match="space"):
        function(helix_3d)


def test_forward_vector_refuses_3d():
    """The forward vector is 2D only and must say so for x-y-z data."""
    position = xr.DataArray(
        np.random.default_rng(0).normal(size=(4, 1, 3, 3)),
        dims=["time", "individual", "keypoint", "space"],
        coords={
            "time": np.arange(4),
            "individual": ["id_0"],
            "keypoint": ["left_ear", "right_ear", "nose"],
            "space": ["x", "y", "z"],
        },
    )

    with pytest.raises(ValueError, match="2 spatial dimensions"):
        kinematics.compute_forward_vector(position, "left_ear", "right_ear")


def test_signed_angle_refuses_3d(helix_3d):
    """The signed angle between two vectors is 2D only."""
    u, v = helix_3d.isel(time=0), helix_3d.isel(time=1)

    with pytest.raises(ValueError, match="space"):
        vector.compute_signed_angle_2d(u, v)
