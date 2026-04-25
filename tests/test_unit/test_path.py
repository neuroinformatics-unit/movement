"""Tests for compute_path_straightness."""

import numpy as np
import pytest
import xarray as xr

from movement.kinematics.path import compute_path_straightness

# ─────────────────────────────────────────────
# Test dataset factories
# ─────────────────────────────────────────────


def _create_test_dataarray(positions, n_ind=1, n_kp=1):
    """Construct a standardized xarray.DataArray for testing."""
    length = len(positions)
    coords = {
        "time": np.arange(length),
        "individuals": [f"id_{i}" for i in range(n_ind)],
        "keypoints": [f"kp_{i}" for i in range(n_kp)],
        "space": ["x", "y"],
    }
    data = np.tile(
        positions[:, np.newaxis, np.newaxis, :], (1, n_ind, n_kp, 1)
    )
    return xr.DataArray(
        data,
        dims=["time", "individuals", "keypoints", "space"],
        coords=coords,
    )


def make_straight_line(length=10, n_ind=1, n_kp=1):
    """Straight diagonal line — SI should be exactly 1.0."""
    t = np.arange(length, dtype=float)
    positions = np.stack([t, t], axis=-1)
    return _create_test_dataarray(positions, n_ind, n_kp)


def make_closed_loop(n_ind=1, n_kp=1):
    """Trajectory returning exactly to start — SI should be 0.0."""
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    return _create_test_dataarray(positions, n_ind, n_kp)


def make_stationary(length=5, n_ind=1, n_kp=1):
    """Animal never moves — path length = 0 → SI should be NaN."""
    positions = np.ones((length, 2)) * 3.0
    return _create_test_dataarray(positions, n_ind, n_kp)


def make_known_si(n_ind=1, n_kp=1):
    """L-shaped path with known SI.

    Path: (0,0) → (3,0) → (3,4)
    D = sqrt(3² + 4²) = 5
    L = 3 + 4 = 7
    SI = 5/7 ≈ 0.7143
    """
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [3.0, 3.0],
            [3.0, 4.0],
        ]
    )
    return _create_test_dataarray(positions, n_ind, n_kp)


# ─────────────────────────────────────────────
# Global straightness tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "data, expected",
    [
        (make_straight_line(length=10), 1.0),
        (make_closed_loop(), 0.0),
        (make_known_si(), 5 / 7),
    ],
)
def test_straightness_known_values(data, expected):
    """SI returns correct value for known trajectories."""
    result = compute_path_straightness(data)
    assert np.allclose(result.values, expected, atol=1e-6)


def test_straightness_stationary_is_nan():
    """Stationary animal has path length 0 → SI should be NaN."""
    result = compute_path_straightness(make_stationary())
    assert np.all(np.isnan(result.values))


def test_straightness_between_zero_and_one():
    """SI must be in [0, 1] for any valid trajectory."""
    result = compute_path_straightness(make_known_si())
    assert np.all((result.values >= 0) & (result.values <= 1))


# ─────────────────────────────────────────────
# NaN handling tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "nan_slice",
    [
        (0, Ellipsis),  # NaN at start
        (-1, Ellipsis),  # NaN at end
        (slice(None), Ellipsis),  # all NaN
    ],
)
def test_straightness_nan_propagates(nan_slice):
    """NaN at start, end, or everywhere propagates to NaN SI."""
    data = make_straight_line(length=10)
    data[nan_slice] = np.nan
    result = compute_path_straightness(data)
    assert np.all(np.isnan(result.values))


# ─────────────────────────────────────────────
# Output shape and metadata tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "n_ind, n_kp",
    [
        (1, 1),
        (2, 3),
        (3, 2),
    ],
)
def test_straightness_output_shape(n_ind, n_kp):
    """SI drops time and space dims, preserves individuals and keypoints."""
    data = make_straight_line(length=10, n_ind=n_ind, n_kp=n_kp)
    result = compute_path_straightness(data)
    assert "time" not in result.dims
    assert "space" not in result.dims
    assert "individuals" in result.dims
    assert "keypoints" in result.dims
    assert result.sizes["individuals"] == n_ind
    assert result.sizes["keypoints"] == n_kp


def test_straightness_output_name_and_units():
    """Output should have correct name and units attributes."""
    result = compute_path_straightness(make_straight_line())
    assert result.name == "straightness_index"
    assert result.attrs.get("units") == "dimensionless"


# ─────────────────────────────────────────────
# Validation tests
# ─────────────────────────────────────────────


def test_straightness_rejects_missing_time_dim():
    """Should raise ValueError when time dimension is absent."""
    data = xr.DataArray(
        np.random.rand(5, 2),
        dims=["individuals", "space"],
        coords={"space": ["x", "y"]},
    )
    with pytest.raises(ValueError):
        compute_path_straightness(data)


def test_straightness_rejects_too_short():
    """Should raise ValueError when fewer than 2 time points."""
    data = make_straight_line(length=1)
    with pytest.raises(ValueError):
        compute_path_straightness(data)
