"""Tests for compute_turning_angles."""

import numpy as np
import pytest
import xarray as xr

from movement.kinematics import compute_turning_angles

# ─────────────────────────────────────────────
# Test dataset factories
# ─────────────────────────────────────────────


def create_straight_line_dataset(length=10, n_ind=1, n_kp=1):
    """Simulate movement in a perfectly straight line (left to right).

    All turning angles should be exactly 0 (no turns).
    """
    coords = {
        "time": np.arange(length),
        "individuals": [f"id_{i}" for i in range(n_ind)],
        "keypoints": [f"kp_{i}" for i in range(n_kp)],
        "space": ["x", "y"],
    }
    # x increases linearly, y stays constant -> heading always 0
    x = np.linspace(0, 9, length)
    y = np.zeros(length)
    base = np.stack([x, y], axis=-1)  # (length, 2)
    data = np.broadcast_to(base, (n_ind, n_kp, length, 2))
    ds = xr.DataArray(
        data.transpose(2, 0, 1, 3),  # (time, individuals, keypoints, space)
        dims=["time", "individuals", "keypoints", "space"],
        coords=coords,
    )
    return ds


def create_right_angle_dataset(n_ind=1, n_kp=1):
    """Simulate a trajectory with a single 90-degree right turn.

    Steps:
        (0,0) -> (1,0) -> (2,0) -> (2,-1) -> (2,-2)

    Headings:    0,      0,      -π/2,    -π/2
    Turning:    NaN,    NaN,     0,       -π/2,     0
    (first two are NaN; first real turn is 0; sharp turn is -π/2)
    """
    positions = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [2, -1],
            [2, -2],
        ],
        dtype=float,
    )
    coords = {
        "time": np.arange(5),
        "individuals": [f"id_{i}" for i in range(n_ind)],
        "keypoints": [f"kp_{i}" for i in range(n_kp)],
        "space": ["x", "y"],
    }
    data = np.broadcast_to(
        positions[:, np.newaxis, np.newaxis, :], (5, n_ind, n_kp, 2)
    )
    return xr.DataArray(
        data,
        dims=["time", "individuals", "keypoints", "space"],
        coords=coords,
    )


def create_left_angle_dataset():
    """Simulate a trajectory with a single 90-degree LEFT turn.

    Steps: (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2)
    Sharp turn should be +π/2.
    """
    positions = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [2, 1],
            [2, 2],
        ],
        dtype=float,
    )
    coords = {
        "time": np.arange(5),
        "individuals": ["id_0"],
        "keypoints": ["kp_0"],
        "space": ["x", "y"],
    }
    data = positions[:, np.newaxis, np.newaxis, :]
    return xr.DataArray(
        data,
        dims=["time", "individuals", "keypoints", "space"],
        coords=coords,
    )


def create_stationary_dataset(length=5, n_ind=1, n_kp=1):
    """All positions identical — animal never moves.

    All turning angles should be NaN (zero-length steps).
    """
    coords = {
        "time": np.arange(length),
        "individuals": [f"id_{i}" for i in range(n_ind)],
        "keypoints": [f"kp_{i}" for i in range(n_kp)],
        "space": ["x", "y"],
    }
    data = np.ones((length, n_ind, n_kp, 2)) * 5.0  # constant position
    return xr.DataArray(
        data,
        dims=["time", "individuals", "keypoints", "space"],
        coords=coords,
    )


def make_nans(dataarray, which="start"):
    """Introduce NaNs at specified locations in the DataArray."""
    ds = dataarray.copy()
    if which == "start":
        ds[0, ...] = np.nan
    elif which == "end":
        ds[-1, ...] = np.nan
    elif which == "all":
        ds[:, ...] = np.nan
    elif which == "middle":
        mid = ds.shape[0] // 2
        ds[mid, ...] = np.nan
    return ds


# ─────────────────────────────────────────────
# Core behaviour tests
# ─────────────────────────────────────────────


def test_turning_angles_straight_line_is_zero():
    """Perfect straight line should have turning angle of 0 everywhere.

    First two time steps are always NaN by design.
    """
    data = create_straight_line_dataset(length=10)
    angles = compute_turning_angles(data)

    # First two time steps must be NaN
    assert np.all(np.isnan(angles.isel(time=0).values))
    assert np.all(np.isnan(angles.isel(time=1).values))

    # Remaining steps: turning angle == 0
    rest = angles.isel(time=slice(2, None))
    assert np.allclose(rest.values, 0.0, atol=1e-10)


def test_turning_angles_right_turn():
    """90-degree right turn should produce -π/2."""
    data = create_right_angle_dataset()
    angles = compute_turning_angles(data)

    # NaN at first two time steps
    assert np.all(np.isnan(angles.isel(time=0).values))
    assert np.all(np.isnan(angles.isel(time=1).values))

    # No turn at step 2 (still going straight)
    assert np.allclose(angles.isel(time=2).values, 0.0, atol=1e-10)

    # Right turn at step 3 should be -π/2
    assert np.allclose(angles.isel(time=3).values, -np.pi / 2, atol=1e-10)

    # Resume straight at step 4
    assert np.allclose(angles.isel(time=4).values, 0.0, atol=1e-10)


def test_turning_angles_left_turn():
    """90-degree left turn should produce +π/2."""
    data = create_left_angle_dataset()
    angles = compute_turning_angles(data)
    assert np.allclose(angles.isel(time=3).values, np.pi / 2, atol=1e-10)


def test_turning_angles_wrapping():
    """A 350-degree raw change should wrap to -10 degrees (~-0.1745 rad)."""
    # Construct a trajectory where heading changes by 350 degrees
    # Heading 1: 0 rad (east), Heading 2: 350 deg = -10 deg -> -π/9
    # Positions: (0,0) -> (1,0) -> (1 + cos(350°), sin(350°))
    angle_350 = np.radians(350)
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0 + np.cos(angle_350), np.sin(angle_350)],
        ]
    )
    coords = {
        "time": np.arange(3),
        "individuals": ["id_0"],
        "keypoints": ["kp_0"],
        "space": ["x", "y"],
    }
    data = xr.DataArray(
        positions[:, np.newaxis, np.newaxis, :],
        dims=["time", "individuals", "keypoints", "space"],
        coords=coords,
    )
    angles = compute_turning_angles(data)
    # 350 degrees raw difference wraps to -10 degrees
    assert np.allclose(angles.isel(time=2).values, np.radians(-10), atol=1e-6)


def test_turning_angles_stationary_all_nan():
    """Stationary animal: zero-length steps -> all NaN turning angles."""
    data = create_stationary_dataset(length=5)
    angles = compute_turning_angles(data)
    assert np.all(np.isnan(angles.values))


# ─────────────────────────────────────────────
# NaN handling tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize("which", ["start", "end", "middle", "all"])
def test_turning_angles_nan_positions(which):
    """NaN positions propagate to NaN turning angles — no crash."""
    data = create_straight_line_dataset(length=10)
    nan_data = make_nans(data, which=which)
    angles = compute_turning_angles(nan_data)

    # Should not raise, should return DataArray
    assert isinstance(angles, xr.DataArray)

    if which == "all":
        assert np.all(np.isnan(angles.values))
    else:
        # At least some NaNs present
        assert np.any(np.isnan(angles.values))


# ─────────────────────────────────────────────
# Output shape and dimension tests
# ─────────────────────────────────────────────


def test_turning_angles_output_shape_preserves_time():
    """Output time dimension must match input time dimension."""
    data = create_straight_line_dataset(length=15, n_ind=2, n_kp=3)
    angles = compute_turning_angles(data)

    assert angles.sizes["time"] == data.sizes["time"]
    assert "individuals" in angles.dims
    assert "keypoints" in angles.dims
    assert "space" not in angles.dims  # space must be dropped


def test_turning_angles_multiple_individuals_keypoints():
    """Function must handle multiple individuals and keypoints correctly."""
    data = create_straight_line_dataset(length=10, n_ind=3, n_kp=2)
    angles = compute_turning_angles(data)

    assert angles.sizes["individuals"] == 3
    assert angles.sizes["keypoints"] == 2
    # All non-NaN values should be zero (straight line)
    valid = angles.isel(time=slice(2, None))
    assert np.allclose(valid.values, 0.0, atol=1e-10)


def test_turning_angles_output_name_and_units():
    """Output DataArray should have name and units attributes set."""
    data = create_straight_line_dataset()
    angles = compute_turning_angles(data)
    assert angles.name == "turning_angle"
    assert angles.attrs.get("units") == "radians"


def test_turning_angles_in_degrees():
    """in_degrees=True should return degrees with correct units attr."""
    data = create_right_angle_dataset()
    angles = compute_turning_angles(data, in_degrees=True)

    assert angles.attrs.get("units") == "degrees"
    # Right turn should be -90 degrees
    assert np.allclose(angles.isel(time=3).values, -90.0, atol=1e-6)


# ─────────────────────────────────────────────
# Validation / error tests
# ─────────────────────────────────────────────


def test_turning_angles_rejects_3d_data():
    """Should raise ValueError for 3D spatial data."""
    coords = {
        "time": np.arange(5),
        "individuals": ["id_0"],
        "keypoints": ["kp_0"],
        "space": ["x", "y", "z"],
    }
    data = xr.DataArray(
        np.random.rand(5, 1, 1, 3),
        dims=["time", "individuals", "keypoints", "space"],
        coords=coords,
    )
    with pytest.raises(ValueError, match="2D"):
        compute_turning_angles(data)


def test_turning_angles_rejects_missing_time_dim():
    """Should raise ValueError when time dimension is absent."""
    data = xr.DataArray(
        np.random.rand(5, 2),
        dims=["individuals", "space"],
        coords={"space": ["x", "y"]},
    )
    with pytest.raises(ValueError):
        compute_turning_angles(data)
