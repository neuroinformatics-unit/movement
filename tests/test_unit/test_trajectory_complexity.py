"""Tests for compute_straightness_index."""

import numpy as np
import pytest
import xarray as xr

from movement.kinematics import compute_straightness_index

# ─────────────────────────────────────────────
# Test dataset factories
# ─────────────────────────────────────────────


def make_straight_line(length=10, n_ind=1, n_kp=1):
    """Straight diagonal line — SI should be exactly 1.0."""
    coords = {
        "time": np.arange(length),
        "individuals": [f"id_{i}" for i in range(n_ind)],
        "keypoints": [f"kp_{i}" for i in range(n_kp)],
        "space": ["x", "y"],
    }
    t = np.arange(length, dtype=float)
    xy = np.stack([t, t], axis=-1)  # diagonal
    data = np.tile(xy[:, np.newaxis, np.newaxis, :], (1, n_ind, n_kp, 1))
    return xr.DataArray(
        data,
        dims=["time", "individuals", "keypoints", "space"],
        coords=coords,
    )


def make_closed_loop(n_ind=1, n_kp=1):
    """Trajectory returning exactly to start — SI should be 0.0."""
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],  # back to start
        ]
    )
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


def make_stationary(length=5, n_ind=1, n_kp=1):
    """Animal never moves — path length = 0 → SI should be NaN."""
    coords = {
        "time": np.arange(length),
        "individuals": [f"id_{i}" for i in range(n_ind)],
        "keypoints": [f"kp_{i}" for i in range(n_kp)],
        "space": ["x", "y"],
    }
    data = np.ones((length, n_ind, n_kp, 2)) * 3.0
    return xr.DataArray(
        data,
        dims=["time", "individuals", "keypoints", "space"],
        coords=coords,
    )


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


# ─────────────────────────────────────────────
# Global straightness tests
# ─────────────────────────────────────────────


def test_straightness_straight_line_is_one():
    """Perfectly straight line should have SI = 1.0."""
    data = make_straight_line(length=10)
    result = compute_straightness_index(data)
    assert np.allclose(result.values, 1.0, atol=1e-10)


def test_straightness_closed_loop_is_zero():
    """Trajectory returning to start should have SI = 0.0."""
    data = make_closed_loop()
    result = compute_straightness_index(data)
    assert np.allclose(result.values, 0.0, atol=1e-10)


def test_straightness_known_value():
    """L-shaped path should return SI = 5/7 ≈ 0.7143."""
    data = make_known_si()
    result = compute_straightness_index(data)
    assert np.allclose(result.values, 5 / 7, atol=1e-6)


def test_straightness_stationary_is_nan():
    """Stationary animal has path length 0 → SI should be NaN."""
    data = make_stationary()
    result = compute_straightness_index(data)
    assert np.all(np.isnan(result.values))


def test_straightness_between_zero_and_one():
    """SI must be in [0, 1] for any valid trajectory."""
    data = make_known_si()
    result = compute_straightness_index(data)
    assert np.all((result.values >= 0) & (result.values <= 1))


# ─────────────────────────────────────────────
# NaN handling tests
# ─────────────────────────────────────────────


def test_straightness_nan_at_start():
    """NaN at start position propagates to NaN SI."""
    data = make_straight_line(length=10)
    data[0, ...] = np.nan
    result = compute_straightness_index(data)
    assert np.all(np.isnan(result.values))


def test_straightness_nan_at_end():
    """NaN at end position propagates to NaN SI."""
    data = make_straight_line(length=10)
    data[-1, ...] = np.nan
    result = compute_straightness_index(data)
    assert np.all(np.isnan(result.values))


def test_straightness_all_nan():
    """All-NaN input should return NaN without crashing."""
    data = make_straight_line(length=10)
    data[:, ...] = np.nan
    result = compute_straightness_index(data)
    assert np.all(np.isnan(result.values))


# ─────────────────────────────────────────────
# Output shape and metadata tests
# ─────────────────────────────────────────────


def test_straightness_drops_space_and_time_dims():
    """Global SI should drop both time and space dimensions."""
    data = make_straight_line(length=10, n_ind=2, n_kp=3)
    result = compute_straightness_index(data)
    assert "time" not in result.dims
    assert "space" not in result.dims
    assert "individuals" in result.dims
    assert "keypoints" in result.dims
    assert result.sizes["individuals"] == 2
    assert result.sizes["keypoints"] == 3


def test_straightness_output_name_and_units():
    """Output should have name and units attributes."""
    data = make_straight_line()
    result = compute_straightness_index(data)
    assert result.name == "straightness_index"
    assert result.attrs.get("units") == "dimensionless"


def test_straightness_multiple_individuals():
    """SI computed correctly for multiple individuals simultaneously."""
    data = make_straight_line(length=10, n_ind=3, n_kp=2)
    result = compute_straightness_index(data)
    assert result.sizes["individuals"] == 3
    assert result.sizes["keypoints"] == 2
    # All straight lines → SI = 1.0 for all
    assert np.allclose(result.values, 1.0, atol=1e-10)


# ─────────────────────────────────────────────
# Rolling window tests
# ─────────────────────────────────────────────


def test_straightness_rolling_output_shape():
    """Rolling SI should preserve time dimension."""
    data = make_straight_line(length=20)
    result = compute_straightness_index(data, window_size=5)
    assert "time" in result.dims
    assert result.sizes["time"] == data.sizes["time"]
    assert "space" not in result.dims


def test_straightness_rolling_first_values_nan():
    """First window_size - 1 values should be NaN (incomplete window)."""
    data = make_straight_line(length=20)
    W = 5
    result = compute_straightness_index(data, window_size=W)
    # First W values should be NaN
    assert np.all(np.isnan(result.isel(time=slice(0, W)).values))


def test_straightness_rolling_straight_line_is_one():
    """Rolling SI on straight line should be 1.0 for all complete windows."""
    data = make_straight_line(length=20)
    W = 5
    result = compute_straightness_index(data, window_size=W)
    valid = result.isel(time=slice(W, None))
    assert np.allclose(valid.values, 1.0, atol=1e-6)


def test_straightness_rolling_invalid_window_size():
    """Negative or zero window_size should raise ValueError."""
    data = make_straight_line()
    with pytest.raises(ValueError):
        compute_straightness_index(data, window_size=0)
    with pytest.raises(ValueError):
        compute_straightness_index(data, window_size=-5)


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
        compute_straightness_index(data)
