import numpy as np
import pytest
import xarray as xr

from movement.kinematics import compute_sinuosity, compute_straightness_index


@pytest.fixture
def clean_zigzag():
    """Return a zigzag trajectory DataArray."""
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
    """Return a straight line trajectory DataArray."""
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
    """Return a zigzag trajectory with a NaN in the middle."""
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
    """Return a zigzag trajectory with multiple NaNs."""
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
    """Return a stationary trajectory DataArray."""
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
    """Return a trajectory DataArray filled with NaNs."""
    time = np.linspace(0, 1, 5)
    space = ["x", "y"]
    data = np.full((5, 2), np.nan)
    return xr.DataArray(
        data,
        coords={"time": time, "space": space},
        dims=["time", "space"],
        name="position",
    )


class TestComputeSinuosity:
    """Tests for the compute_sinuosity function."""

    def test_straight_line_sinuosity_is_one(self, straight_line):
        """Test that sinuosity of a straight line is 1.0."""
        result = compute_sinuosity(straight_line)
        assert np.isclose(result.values, 1.0, atol=1e-6)

    def test_zigzag_sinuosity_greater_than_one(self, clean_zigzag):
        """Test that zigzag sinuosity is greater than 1.0."""
        result = compute_sinuosity(clean_zigzag)
        assert result.values > 1.0

    def test_zigzag_sinuosity_expected_value(self, clean_zigzag):
        """Test that zigzag sinuosity matches the expected value."""
        result = compute_sinuosity(clean_zigzag)
        expected = np.sqrt(2)
        assert np.isclose(result.values, expected, atol=1e-4)

    def test_stationary_returns_nan(self, stationary):
        """Test that stationary trajectories return NaN."""
        result = compute_sinuosity(stationary)
        assert np.isnan(result.values)

    def test_all_nan_returns_nan(self, all_nan):
        """Test that all-NaN trajectories return NaN."""
        result = compute_sinuosity(all_nan)
        assert np.isnan(result.values)

    @pytest.mark.parametrize("nan_policy", ["ffill", "scale"])
    def test_nan_trajectory_computes_with_policy(
        self, zigzag_with_nan, nan_policy
    ):
        """Test computation for NaN data using valid policies."""
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
        """Test computation with multiple NaNs using valid policies."""
        result = compute_sinuosity(
            zigzag_multi_nan,
            nan_policy=nan_policy,
            nan_warn_threshold=0.5,
        )
        assert not np.isnan(result.values)
        assert result.values > 1.0

    def test_output_is_dataarray(self, clean_zigzag):
        """Test that the output is an xarray DataArray."""
        result = compute_sinuosity(clean_zigzag)
        assert isinstance(result, xr.DataArray)

    def test_output_name(self, clean_zigzag):
        """Test that the output DataArray is named 'sinuosity'."""
        result = compute_sinuosity(clean_zigzag)
        assert result.name == "sinuosity"


class TestComputeStraightnessIndex:
    """Tests for the compute_straightness_index function."""

    def test_straight_line_straightness_is_one(self, straight_line):
        """Test that the straightness index of a straight line is 1.0."""
        result = compute_straightness_index(straight_line)
        assert np.isclose(result.values, 1.0, atol=1e-6)

    def test_zigzag_straightness_less_than_one(self, clean_zigzag):
        """Test that zigzag straightness index is less than 1.0."""
        result = compute_straightness_index(clean_zigzag)
        assert result.values < 1.0

    def test_zigzag_straightness_expected_value(self, clean_zigzag):
        """Test that zigzag straightness matches the expected value."""
        result = compute_straightness_index(clean_zigzag)
        expected = 1.0 / np.sqrt(2)
        assert np.isclose(result.values, expected, atol=1e-4)

    def test_stationary_returns_nan(self, stationary):
        """Test that stationary trajectories return NaN."""
        result = compute_straightness_index(stationary)
        assert np.isnan(result.values)

    def test_all_nan_returns_nan(self, all_nan):
        """Test that all-NaN trajectories return NaN."""
        result = compute_straightness_index(all_nan)
        assert np.isnan(result.values)

    @pytest.mark.parametrize("nan_policy", ["ffill", "scale"])
    def test_nan_trajectory_computes_with_policy(
        self, zigzag_with_nan, nan_policy
    ):
        """Test straightness for NaN data with valid policies."""
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
        """Test straightness for multiple NaNs with valid policies."""
        result = compute_straightness_index(
            zigzag_multi_nan,
            nan_policy=nan_policy,
            nan_warn_threshold=0.5,
        )
        assert not np.isnan(result.values)
        assert 0 < result.values <= 1.0

    def test_output_is_dataarray(self, clean_zigzag):
        """Test that the output is an xarray DataArray."""
        result = compute_straightness_index(clean_zigzag)
        assert isinstance(result, xr.DataArray)

    def test_output_name(self, clean_zigzag):
        """Test that output is named 'straightness_index'."""
        result = compute_straightness_index(clean_zigzag)
        assert result.name == "straightness_index"


class TestSinuosityStraightnessRelationship:
    """Tests for the relationship between sinuosity and straightness."""

    def test_reciprocal_relationship(self, clean_zigzag):
        """Test that sinuosity and straightness are reciprocal."""
        sinuosity = compute_sinuosity(clean_zigzag)
        straightness = compute_straightness_index(clean_zigzag)
        product = sinuosity.values * straightness.values
        assert np.isclose(product, 1.0, atol=1e-6)

    def test_reciprocal_with_straight_line(self, straight_line):
        """Test the reciprocal relationship for a straight line."""
        sinuosity = compute_sinuosity(straight_line)
        straightness = compute_straightness_index(straight_line)
        assert np.isclose(sinuosity.values, 1.0, atol=1e-6)
        assert np.isclose(straightness.values, 1.0, atol=1e-6)

    @pytest.mark.parametrize("nan_policy", ["ffill", "scale"])
    def test_reciprocal_with_nan_data(self, zigzag_with_nan, nan_policy):
        """Test the reciprocal relationship for NaN data."""
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
        """Test that both indices are NaN for stationary data."""
        sinuosity = compute_sinuosity(stationary)
        straightness = compute_straightness_index(stationary)
        assert np.isnan(sinuosity.values)
        assert np.isnan(straightness.values)
