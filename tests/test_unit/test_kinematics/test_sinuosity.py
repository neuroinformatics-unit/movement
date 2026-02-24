"""Unit tests for compute_sinuosity and compute_straightness_index."""

import numpy as np
import pytest
import xarray as xr

from movement.kinematics import compute_sinuosity, compute_straightness_index

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def clean_zigzag():
    """Create a clean zig-zag trajectory with no NaN values."""
    time = np.linspace(0, 1, 5)
    space = ["x", "y"]
    data = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]], dtype=float)
    return xr.DataArray(
        data,
        coords={"time": time, "space": space},
        dims=["time", "space"],
        name="position",
    )


@pytest.fixture
def straight_line():
    """Create a perfectly straight trajectory along the x-axis."""
    time = np.linspace(0, 1, 5)
    space = ["x", "y"]
    data = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
    return xr.DataArray(
        data,
        coords={"time": time, "space": space},
        dims=["time", "space"],
        name="position",
    )


@pytest.fixture
def zigzag_with_nan():
    """Create a zig-zag trajectory with a single NaN gap."""
    time = np.linspace(0, 1, 5)
    space = ["x", "y"]
    data = np.array(
        [[0, 0], [1, 1], [np.nan, np.nan], [3, 1], [4, 0]], dtype=float
    )
    return xr.DataArray(
        data,
        coords={"time": time, "space": space},
        dims=["time", "space"],
        name="position",
    )


@pytest.fixture
def zigzag_multi_nan():
    """Create a trajectory with multiple consecutive NaN gaps."""
    time = np.linspace(0, 1, 9)
    space = ["x", "y"]
    data = np.array(
        [
            [0, 0],
            [1, 1],
            [np.nan, np.nan],
            [3, 1],
            [4, 0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [7, 1],
            [8, 0],
        ],
        dtype=float,
    )
    return xr.DataArray(
        data,
        coords={"time": time, "space": space},
        dims=["time", "space"],
        name="position",
    )


@pytest.fixture
def stationary():
    """Create a stationary trajectory (no movement)."""
    time = np.linspace(0, 1, 5)
    space = ["x", "y"]
    data = np.array([[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]], dtype=float)
    return xr.DataArray(
        data,
        coords={"time": time, "space": space},
        dims=["time", "space"],
        name="position",
    )


@pytest.fixture
def all_nan():
    """Create a trajectory where all points are NaN."""
    time = np.linspace(0, 1, 5)
    space = ["x", "y"]
    data = np.full((5, 2), np.nan)
    return xr.DataArray(
        data,
        coords={"time": time, "space": space},
        dims=["time", "space"],
        name="position",
    )


# ============================================================
# Tests for compute_sinuosity
# ============================================================


class TestComputeSinuosity:
    """Tests for compute_sinuosity."""

    def test_straight_line_sinuosity_is_one(self, straight_line):
        """A perfectly straight path should have sinuosity of 1."""
        result = compute_sinuosity(straight_line)
        assert np.isclose(result.values, 1.0, atol=1e-6)

    def test_zigzag_sinuosity_greater_than_one(self, clean_zigzag):
        """A zig-zag path should have sinuosity > 1."""
        result = compute_sinuosity(clean_zigzag)
        assert result.values > 1.0

    def test_zigzag_sinuosity_expected_value(self, clean_zigzag):
        """Verify the exact sinuosity of the known zig-zag trajectory.

        Path length = 4 * sqrt(2), net displacement = 4.
        Sinuosity = 4*sqrt(2) / 4 = sqrt(2) ≈ 1.4142.
        """
        result = compute_sinuosity(clean_zigzag)
        expected = np.sqrt(2)
        assert np.isclose(result.values, expected, atol=1e-4)

    def test_stationary_returns_nan(self, stationary):
        """A stationary trajectory should return NaN silently."""
        result = compute_sinuosity(stationary)
        assert np.isnan(result.values)

    def test_all_nan_returns_nan(self, all_nan):
        """An all-NaN trajectory should return NaN."""
        result = compute_sinuosity(all_nan)
        assert np.isnan(result.values)

    @pytest.mark.parametrize("nan_policy", ["ffill", "scale"])
    def test_nan_trajectory_computes_with_policy(
        self, zigzag_with_nan, nan_policy
    ):
        """Sinuosity should be computable for NaN data with valid policies."""
        result = compute_sinuosity(
            zigzag_with_nan,
            nan_policy=nan_policy,
            nan_warn_threshold=0.5,
        )
        assert not np.isnan(result.values)
        assert result.values > 1.0

    @pytest.mark.parametrize("nan_policy", ["ffill", "scale"])
    def test_multi_nan_computes_with_policy(
        self, zigzag_multi_nan, nan_policy
    ):
        """Sinuosity should handle multiple NaN gaps gracefully."""
        result = compute_sinuosity(
            zigzag_multi_nan,
            nan_policy=nan_policy,
            nan_warn_threshold=0.5,
        )
        assert not np.isnan(result.values)
        assert result.values > 1.0

    def test_output_is_dataarray(self, clean_zigzag):
        """Output must be an xr.DataArray."""
        result = compute_sinuosity(clean_zigzag)
        assert isinstance(result, xr.DataArray)

    def test_output_name(self, clean_zigzag):
        """Output DataArray should be named 'sinuosity'."""
        result = compute_sinuosity(clean_zigzag)
        assert result.name == "sinuosity"


# ============================================================
# Tests for compute_straightness_index
# ============================================================


class TestComputeStraightnessIndex:
    """Tests for compute_straightness_index."""

    def test_straight_line_straightness_is_one(self, straight_line):
        """A perfectly straight path should have straightness of 1."""
        result = compute_straightness_index(straight_line)
        assert np.isclose(result.values, 1.0, atol=1e-6)

    def test_zigzag_straightness_less_than_one(self, clean_zigzag):
        """A zig-zag path should have straightness < 1."""
        result = compute_straightness_index(clean_zigzag)
        assert result.values < 1.0

    def test_zigzag_straightness_expected_value(self, clean_zigzag):
        """Verify the exact straightness of the known zig-zag trajectory.

        Net displacement = 4, path length = 4 * sqrt(2).
        Straightness = 4 / (4*sqrt(2)) = 1/sqrt(2) ≈ 0.7071.
        """
        result = compute_straightness_index(clean_zigzag)
        expected = 1.0 / np.sqrt(2)
        assert np.isclose(result.values, expected, atol=1e-4)

    def test_stationary_returns_nan(self, stationary):
        """A stationary trajectory should return NaN silently."""
        result = compute_straightness_index(stationary)
        assert np.isnan(result.values)

    def test_all_nan_returns_nan(self, all_nan):
        """An all-NaN trajectory should return NaN."""
        result = compute_straightness_index(all_nan)
        assert np.isnan(result.values)

    @pytest.mark.parametrize("nan_policy", ["ffill", "scale"])
    def test_nan_trajectory_computes_with_policy(
        self, zigzag_with_nan, nan_policy
    ):
        """Straightness should be computable for NaN data with valid policies."""
        result = compute_straightness_index(
            zigzag_with_nan,
            nan_policy=nan_policy,
            nan_warn_threshold=0.5,
        )
        assert not np.isnan(result.values)
        assert 0 < result.values <= 1.0

    @pytest.mark.parametrize("nan_policy", ["ffill", "scale"])
    def test_multi_nan_computes_with_policy(
        self, zigzag_multi_nan, nan_policy
    ):
        """Straightness should handle multiple NaN gaps gracefully."""
        result = compute_straightness_index(
            zigzag_multi_nan,
            nan_policy=nan_policy,
            nan_warn_threshold=0.5,
        )
        assert not np.isnan(result.values)
        assert 0 < result.values <= 1.0

    def test_output_is_dataarray(self, clean_zigzag):
        """Output must be an xr.DataArray."""
        result = compute_straightness_index(clean_zigzag)
        assert isinstance(result, xr.DataArray)

    def test_output_name(self, clean_zigzag):
        """Output DataArray should be named 'straightness_index'."""
        result = compute_straightness_index(clean_zigzag)
        assert result.name == "straightness_index"


# ============================================================
# Cross-validation tests
# ============================================================


class TestSinuosityStraightnessRelationship:
    """Tests verifying the mathematical relationship between metrics."""

    def test_reciprocal_relationship(self, clean_zigzag):
        """Sinuosity * Straightness should equal 1."""
        sinuosity = compute_sinuosity(clean_zigzag)
        straightness = compute_straightness_index(clean_zigzag)
        product = sinuosity.values * straightness.values
        assert np.isclose(product, 1.0, atol=1e-6)

    def test_reciprocal_with_straight_line(self, straight_line):
        """Both metrics should be 1 for a straight path."""
        sinuosity = compute_sinuosity(straight_line)
        straightness = compute_straightness_index(straight_line)
        assert np.isclose(sinuosity.values, 1.0, atol=1e-6)
        assert np.isclose(straightness.values, 1.0, atol=1e-6)

    @pytest.mark.parametrize("nan_policy", ["ffill", "scale"])
    def test_reciprocal_with_nan_data(self, zigzag_with_nan, nan_policy):
        """Reciprocal relationship should hold even with NaN data."""
        sinuosity = compute_sinuosity(
            zigzag_with_nan,
            nan_policy=nan_policy,
            nan_warn_threshold=0.5,
        )
        straightness = compute_straightness_index(
            zigzag_with_nan,
            nan_policy=nan_policy,
            nan_warn_threshold=0.5,
        )
        product = sinuosity.values * straightness.values
        assert np.isclose(product, 1.0, atol=1e-6)

    def test_both_nan_for_stationary(self, stationary):
        """Both metrics should return NaN for stationary trajectories."""
        sinuosity = compute_sinuosity(stationary)
        straightness = compute_straightness_index(stationary)
        assert np.isnan(sinuosity.values)
        assert np.isnan(straightness.values)
