"""Tests for the collective behavior metrics module."""

import numpy as np
import pytest
import xarray as xr

from movement import kinematics


@pytest.fixture
def position_data_aligned_individuals():
    """Return position data for 3 individuals all moving in the same direction.

    All individuals move along the positive x-axis at every time step,
    so polarization should be 1.0.
    """
    time = [0, 1, 2, 3]
    individuals = ["id_0", "id_1", "id_2"]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # All individuals move in +x direction
    # Shape: (time=4, space=2, keypoints=1, individuals=3)
    # x-coords: all increase by 1 each time step
    # y-coords: all stay at 0
    data = np.array(
        [
            # time 0: x=[0,1,2], y=[0,0,0]
            [[[0, 1, 2]], [[0, 0, 0]]],
            # time 1: x=[1,2,3], y=[0,0,0]
            [[[1, 2, 3]], [[0, 0, 0]]],
            # time 2: x=[2,3,4], y=[0,0,0]
            [[[2, 3, 4]], [[0, 0, 0]]],
            # time 3: x=[3,4,5], y=[0,0,0]
            [[[3, 4, 5]], [[0, 0, 0]]],
        ],
        dtype=float,
    )

    return xr.DataArray(
        data,
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": time,
            "space": space,
            "keypoints": keypoints,
            "individuals": individuals,
        },
    )


@pytest.fixture
def position_data_opposite_individuals():
    """Return position data for 2 individuals moving in opposite directions.

    One moves in +x, the other in -x, so polarization should be 0.0.
    """
    time = [0, 1, 2, 3]
    individuals = ["id_0", "id_1"]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # id_0 moves in +x, id_1 moves in -x
    # Shape: (time=4, space=2, keypoints=1, individuals=2)
    data = np.array(
        [
            # time 0: x=[0,10], y=[0,0]
            [[[0, 10]], [[0, 0]]],
            # time 1: x=[1,9], y=[0,0]
            [[[1, 9]], [[0, 0]]],
            # time 2: x=[2,8], y=[0,0]
            [[[2, 8]], [[0, 0]]],
            # time 3: x=[3,7], y=[0,0]
            [[[3, 7]], [[0, 0]]],
        ],
        dtype=float,
    )

    return xr.DataArray(
        data,
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": time,
            "space": space,
            "keypoints": keypoints,
            "individuals": individuals,
        },
    )


@pytest.fixture
def position_data_with_keypoints():
    """Return position data with origin/target keypoints for heading.

    Two individuals, both facing the same direction (positive x).
    Heading is computed as tail -> nose (origin -> target).
    """
    time = [0, 1, 2]
    individuals = ["id_0", "id_1"]
    keypoints = ["nose", "tail"]
    space = ["x", "y"]

    # Both individuals facing +x (nose ahead of tail in x)
    # Shape: (time=3, space=2, keypoints=2, individuals=2)
    # For each individual: nose is at higher x than tail
    data = np.array(
        [
            # time 0: nose_x=[2,5], nose_y=[0,1], tail_x=[0,3], tail_y=[0,1]
            [[[2, 5], [0, 3]], [[0, 1], [0, 1]]],
            # time 1
            [[[3, 6], [1, 4]], [[0, 1], [0, 1]]],
            # time 2
            [[[4, 7], [2, 5]], [[0, 1], [0, 1]]],
        ],
        dtype=float,
    )

    return xr.DataArray(
        data,
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": time,
            "space": space,
            "keypoints": keypoints,
            "individuals": individuals,
        },
    )


@pytest.fixture
def position_data_with_nan():
    """Return position data with NaN values for one individual at one time."""
    time = [0, 1, 2]
    individuals = ["id_0", "id_1", "id_2"]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # Shape: (time=3, space=2, keypoints=1, individuals=3)
    # id_1 has NaN at time 1
    data = np.array(
        [
            # time 0: x=[0,1,2], y=[0,0,0] - all valid
            [[[0, 1, 2]], [[0, 0, 0]]],
            # time 1: x=[1,nan,3], y=[0,nan,0] - id_1 is NaN
            [[[1, np.nan, 3]], [[0, np.nan, 0]]],
            # time 2: x=[2,3,4], y=[0,0,0] - all valid
            [[[2, 3, 4]], [[0, 0, 0]]],
        ],
        dtype=float,
    )

    return xr.DataArray(
        data,
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": time,
            "space": space,
            "keypoints": keypoints,
            "individuals": individuals,
        },
    )


class TestComputePolarization:
    """Test suite for the compute_polarization function."""

    def test_polarization_aligned(self, position_data_aligned_individuals):
        """Test polarization is 1.0 when all move same direction."""
        polarization = kinematics.compute_polarization(
            position_data_aligned_individuals
        )

        assert isinstance(polarization, xr.DataArray)
        assert polarization.name == "polarization"
        assert "time" in polarization.dims
        assert "individuals" not in polarization.dims
        assert "space" not in polarization.dims

        # All moving in same direction -> polarization should be ~1.0
        # (Skip first time point since velocity is computed via diff)
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)

    def test_polarization_opposite(self, position_data_opposite_individuals):
        """Test polarization is 0.0 when individuals move opposite."""
        polarization = kinematics.compute_polarization(
            position_data_opposite_individuals
        )

        assert isinstance(polarization, xr.DataArray)
        assert polarization.name == "polarization"

        # Opposite directions -> polarization should be ~0.0
        # (Skip first time point since velocity is computed via diff)
        assert np.allclose(polarization.values[1:], 0.0, atol=1e-10)

    def test_polarization_with_keypoints(self, position_data_with_keypoints):
        """Test polarization using keypoint-based heading."""
        polarization = kinematics.compute_polarization(
            position_data_with_keypoints,
            heading_keypoints=("tail", "nose"),  # origin -> target
        )

        assert isinstance(polarization, xr.DataArray)
        assert polarization.name == "polarization"

        # Both facing same direction -> polarization should be 1.0
        assert np.allclose(polarization.values, 1.0, atol=1e-10)

    def test_polarization_handles_nan(self, position_data_with_nan):
        """Test that NaN values are handled correctly."""
        polarization = kinematics.compute_polarization(position_data_with_nan)

        assert isinstance(polarization, xr.DataArray)
        # Should compute polarization even with missing data
        # The frame with NaN should exclude that individual from calculation
        assert not np.all(np.isnan(polarization.values))

    def test_polarization_range(self, position_data_aligned_individuals):
        """Test that polarization values are in [0, 1] range."""
        polarization = kinematics.compute_polarization(
            position_data_aligned_individuals
        )

        # Exclude NaN values from range check
        valid_values = polarization.values[~np.isnan(polarization.values)]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 1.0)

    def test_invalid_input_type(self, position_data_aligned_individuals):
        """Test that non-DataArray input raises TypeError."""
        with pytest.raises(TypeError, match="must be an xarray.DataArray"):
            kinematics.compute_polarization(
                position_data_aligned_individuals.values
            )

    def test_missing_dimensions(self, position_data_aligned_individuals):
        """Test that missing required dimensions raises ValueError."""
        # Drop individuals dimension
        data_no_individuals = position_data_aligned_individuals.sel(
            individuals="id_0", drop=True
        )
        with pytest.raises(ValueError, match="individuals"):
            kinematics.compute_polarization(data_no_individuals)

    def test_invalid_keypoints(self, position_data_with_keypoints):
        """Test that invalid keypoint names raise ValueError."""
        with pytest.raises(ValueError, match="nonexistent"):
            kinematics.compute_polarization(
                position_data_with_keypoints,
                heading_keypoints=("nose", "nonexistent"),
            )

    def test_identical_keypoints(self, position_data_with_keypoints):
        """Test that identical origin and target keypoints raise ValueError."""
        with pytest.raises(ValueError, match="may not be identical"):
            kinematics.compute_polarization(
                position_data_with_keypoints,
                heading_keypoints=("nose", "nose"),
            )
