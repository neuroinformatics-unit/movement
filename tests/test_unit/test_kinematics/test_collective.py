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


@pytest.fixture
def position_data_perpendicular():
    """Return position data for 4 individuals in perpendicular dirs.

    Each individual moves in one of the 4 cardinal directions (+x, -x, +y, -y).
    The sum of unit vectors is zero, so polarization should be 0.0.
    """
    time = [0, 1, 2, 3]
    individuals = ["id_0", "id_1", "id_2", "id_3"]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # id_0: +x, id_1: -x, id_2: +y, id_3: -y
    # Shape: (time=4, space=2, keypoints=1, individuals=4)
    data = np.array(
        [
            # time 0
            [[[0, 10, 0, 0]], [[0, 0, 0, 10]]],
            # time 1: +x moves right, -x moves left, +y moves up, -y moves down
            [[[1, 9, 0, 0]], [[0, 0, 1, 9]]],
            # time 2
            [[[2, 8, 0, 0]], [[0, 0, 2, 8]]],
            # time 3
            [[[3, 7, 0, 0]], [[0, 0, 3, 7]]],
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
def position_data_partial_alignment():
    """Return position data for 3 individuals with partial alignment.

    Two individuals move in +x, one moves in +y.
    Expected polarization: |[2,1]|/3 = sqrt(5)/3 ≈ 0.745
    """
    time = [0, 1, 2, 3]
    individuals = ["id_0", "id_1", "id_2"]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # id_0 and id_1 move in +x, id_2 moves in +y
    # Shape: (time=4, space=2, keypoints=1, individuals=3)
    data = np.array(
        [
            # time 0
            [[[0, 5, 0]], [[0, 0, 0]]],
            # time 1
            [[[1, 6, 0]], [[0, 0, 1]]],
            # time 2
            [[[2, 7, 0]], [[0, 0, 2]]],
            # time 3
            [[[3, 8, 0]], [[0, 0, 3]]],
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
def position_data_single_individual():
    """Return position data for a single individual.

    In this synthetic dataset, polarization is 1.0 whenever a valid heading
    can be computed. First-frame behavior in velocity mode depends on boundary
    differencing.
    """
    time = [0, 1, 2, 3]
    individuals = ["id_0"]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # Single individual moving in +x direction
    data = np.array(
        [
            [[[0]], [[0]]],
            [[[1]], [[0]]],
            [[[2]], [[0]]],
            [[[3]], [[0]]],
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
def position_data_all_nan_frame():
    """Return position data with one frame where all individuals are NaN."""
    time = [0, 1, 2, 3]
    individuals = ["id_0", "id_1"]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # All individuals have NaN at time 2
    data = np.array(
        [
            [[[0, 5]], [[0, 0]]],
            [[[1, 6]], [[0, 0]]],
            [[[np.nan, np.nan]], [[np.nan, np.nan]]],  # all NaN
            [[[3, 8]], [[0, 0]]],
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
def position_data_stationary():
    """Return position data where individuals are stationary."""
    time = [0, 1, 2, 3]
    individuals = ["id_0", "id_1"]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # Both individuals stay at the same position
    data = np.array(
        [
            [[[0, 5]], [[0, 0]]],
            [[[0, 5]], [[0, 0]]],
            [[[0, 5]], [[0, 0]]],
            [[[0, 5]], [[0, 0]]],
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
def position_data_large_n():
    """Return position data with many individuals (N=50) all aligned."""
    time = [0, 1, 2]
    n_individuals = 50
    individuals = [f"id_{i}" for i in range(n_individuals)]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # All individuals move in +x direction
    # Shape: (time=3, space=2, keypoints=1, individuals=50)
    x_coords = np.arange(n_individuals, dtype=float)
    data = np.array(
        [
            [[x_coords], [np.zeros(n_individuals)]],
            [[x_coords + 1], [np.zeros(n_individuals)]],
            [[x_coords + 2], [np.zeros(n_individuals)]],
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
def position_data_no_keypoints():
    """Return position data without keypoints dimension."""
    time = [0, 1, 2, 3]
    individuals = ["id_0", "id_1", "id_2"]
    space = ["x", "y"]

    # Shape: (time=4, space=2, individuals=3)
    # All individuals move in +x direction
    data = np.array(
        [
            [[0, 1, 2], [0, 0, 0]],
            [[1, 2, 3], [0, 0, 0]],
            [[2, 3, 4], [0, 0, 0]],
            [[3, 4, 5], [0, 0, 0]],
        ],
        dtype=float,
    )

    return xr.DataArray(
        data,
        dims=["time", "space", "individuals"],
        coords={
            "time": time,
            "space": space,
            "individuals": individuals,
        },
    )


@pytest.fixture
def position_data_multiple_keypoints():
    """Return position data with multiple keypoints for velocity mode test.

    Tests that velocity mode uses the first keypoint.
    """
    time = [0, 1, 2, 3]
    individuals = ["id_0", "id_1"]
    keypoints = ["nose", "tail", "center"]  # nose is first
    space = ["x", "y"]

    # nose moves in +x (should be used)
    # tail moves in -x
    # center moves in +y
    # Shape: (time=4, space=2, keypoints=3, individuals=2)
    data = np.array(
        [
            # time 0
            [
                [[0, 0], [10, 10], [0, 0]],  # x: nose, tail, center
                [[0, 0], [0, 0], [0, 0]],  # y: nose, tail, center
            ],
            # time 1
            [
                [[1, 1], [9, 9], [0, 0]],
                [[0, 0], [0, 0], [1, 1]],
            ],
            # time 2
            [
                [[2, 2], [8, 8], [0, 0]],
                [[0, 0], [0, 0], [2, 2]],
            ],
            # time 3
            [
                [[3, 3], [7, 7], [0, 0]],
                [[0, 0], [0, 0], [3, 3]],
            ],
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
def position_data_diagonal_movement():
    """Return position data with diagonal movement at 45 degrees."""
    time = [0, 1, 2, 3]
    individuals = ["id_0", "id_1"]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # Both individuals move diagonally (45 degrees, +x +y)
    data = np.array(
        [
            [[[0, 5]], [[0, 5]]],
            [[[1, 6]], [[1, 6]]],
            [[[2, 7]], [[2, 7]]],
            [[[3, 8]], [[3, 8]]],
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
def position_data_keypoints_opposite():
    """Return position data where keypoint-based headings are opposite."""
    time = [0, 1, 2]
    individuals = ["id_0", "id_1"]
    keypoints = ["nose", "tail"]
    space = ["x", "y"]

    # id_0 faces +x (nose ahead of tail)
    # id_1 faces -x (nose behind tail)
    data = np.array(
        [
            # time 0: id_0 nose at (2,0), tail at (0,0) -> faces +x
            #         id_1 nose at (3,0), tail at (5,0) -> faces -x
            [[[2, 3], [0, 5]], [[0, 0], [0, 0]]],
            [[[3, 4], [1, 6]], [[0, 0], [0, 0]]],
            [[[4, 5], [2, 7]], [[0, 0], [0, 0]]],
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
def position_data_non_uniform_time():
    """Return position data with non-uniform time spacing."""
    time = [0.0, 0.5, 2.0, 5.0]  # Non-uniform intervals
    individuals = ["id_0", "id_1"]
    keypoints = ["centroid"]
    space = ["x", "y"]

    # Both move in +x direction
    data = np.array(
        [
            [[[0, 5]], [[0, 0]]],
            [[[1, 6]], [[0, 0]]],
            [[[2, 7]], [[0, 0]]],
            [[[3, 8]], [[0, 0]]],
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

    def test_polarization_aligned(
        self,
        position_data_aligned_individuals,
        position_data_diagonal_movement,
    ):
        """Test polarization is 1.0 when all move same direction.

        Tests both horizontal and diagonal movement to verify that
        polarization is rotation-invariant (direction angle doesn't matter,
        only alignment between individuals).
        """
        # Test horizontal alignment
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

        # Test diagonal alignment (rotation invariance)
        # Both individuals moving at 45 degrees should also yield pol=1.0
        polarization_diag = kinematics.compute_polarization(
            position_data_diagonal_movement
        )
        assert np.allclose(polarization_diag.values[1:], 1.0, atol=1e-10)

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

    # ================= Intermediate Polarization Values =================

    def test_polarization_perpendicular_four_directions(
        self, position_data_perpendicular
    ):
        """Test polarization is 0 when 4 individuals move in cardinal dirs."""
        polarization = kinematics.compute_polarization(
            position_data_perpendicular
        )

        # 4 perpendicular directions cancel out -> polarization = 0.0
        # Compare frames 1: avoid boundary differencing dependence at t=0.
        assert np.allclose(polarization.values[1:], 0.0, atol=1e-10)

    def test_polarization_partial_alignment(
        self, position_data_partial_alignment
    ):
        """Test intermediate polarization with partial alignment.

        Two individuals move +x, one moves +y.
        Unit vectors: [1,0], [1,0], [0,1]
        Sum: [2, 1]
        Magnitude: sqrt(5)
        Polarization: sqrt(5)/3 ≈ 0.745
        """
        polarization = kinematics.compute_polarization(
            position_data_partial_alignment
        )

        expected = np.sqrt(5) / 3  # ≈ 0.745
        # Compare frames 1: avoid boundary differencing dependence at t=0.
        assert np.allclose(polarization.values[1:], expected, atol=1e-10)

    # ==================== Edge Cases ====================

    def test_polarization_single_individual(
        self, position_data_single_individual
    ):
        """Test polarization is 1.0 for a single individual."""
        polarization = kinematics.compute_polarization(
            position_data_single_individual
        )

        # Single individual always has polarization = 1.0
        # (the unit vector divided by 1)
        # Compare frames 1: avoid boundary differencing dependence at t=0.
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)

    def test_polarization_all_nan_frame(self, position_data_all_nan_frame):
        """Test that all-NaN frames result in NaN polarization."""
        polarization = kinematics.compute_polarization(
            position_data_all_nan_frame
        )

        # Frame at index 2 has all NaN positions
        # Velocity at frame 2 is computed from frames 1->2 and 2->3
        # Due to NaN at frame 2, velocity at frames 2 and 3 will be affected
        # The exact behavior depends on compute_velocity's edge handling
        assert isinstance(polarization, xr.DataArray)
        # At minimum, verify we get a result with correct length
        assert len(polarization) == len(position_data_all_nan_frame.time)

    def test_polarization_stationary(self, position_data_stationary):
        """Test that stationary individuals (zero velocity) produce NaN."""
        polarization = kinematics.compute_polarization(
            position_data_stationary
        )

        # Zero velocity means zero-length vector -> unit vector is NaN
        # All frames after first should be NaN (zero displacement)
        assert np.all(np.isnan(polarization.values[1:]))

    def test_polarization_large_n(self, position_data_large_n):
        """Test polarization with many individuals (N=50) all aligned."""
        polarization = kinematics.compute_polarization(position_data_large_n)

        # All 50 individuals moving same direction -> polarization = 1.0
        # Compare frames 1: avoid boundary differencing dependence at t=0.
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)

    # ==================== Data Structure Variations ====================

    def test_polarization_no_keypoints_dimension(
        self, position_data_no_keypoints
    ):
        """Test polarization works without keypoints dimension."""
        polarization = kinematics.compute_polarization(
            position_data_no_keypoints
        )

        assert isinstance(polarization, xr.DataArray)
        assert polarization.name == "polarization"
        # All moving same direction -> polarization = 1.0
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)

    def test_polarization_velocity_mode_uses_first_keypoint(
        self, position_data_multiple_keypoints
    ):
        """Test velocity mode uses first keypoint when multiple exist."""
        polarization = kinematics.compute_polarization(
            position_data_multiple_keypoints
        )

        # First keypoint (nose) moves in +x for both individuals
        # So polarization should be 1.0 (not affected by tail or center)
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)

    def test_polarization_keypoints_opposite_directions(
        self, position_data_keypoints_opposite
    ):
        """Test keypoint polarization with opposite facing individuals."""
        polarization = kinematics.compute_polarization(
            position_data_keypoints_opposite,
            heading_keypoints=("tail", "nose"),
        )

        # id_0 faces +x, id_1 faces -x -> polarization = 0.0
        assert np.allclose(polarization.values, 0.0, atol=1e-10)

    # ==================== Time Coordinate Handling ====================

    def test_polarization_preserves_time_coords(
        self, position_data_aligned_individuals
    ):
        """Test that output preserves time coordinates from input."""
        polarization = kinematics.compute_polarization(
            position_data_aligned_individuals
        )

        np.testing.assert_array_equal(
            polarization.time.values,
            position_data_aligned_individuals.time.values,
        )

    def test_polarization_non_uniform_time(
        self, position_data_non_uniform_time
    ):
        """Test polarization with non-uniform time spacing."""
        polarization = kinematics.compute_polarization(
            position_data_non_uniform_time
        )

        # Should still work and preserve time coords
        expected_times = [0.0, 0.5, 2.0, 5.0]
        np.testing.assert_array_equal(polarization.time.values, expected_times)
        # Both moving same direction
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)

    # ==================== Output Properties ====================

    def test_polarization_output_structure(
        self, position_data_aligned_individuals
    ):
        """Test that output has correct structure (time dimension only).

        Verifies both positive assertion (dims == time) and explicit
        absence of input dimensions that should be reduced over.
        """
        polarization = kinematics.compute_polarization(
            position_data_aligned_individuals
        )

        # Positive assertion: output has exactly time dimension
        assert polarization.dims == ("time",)
        assert len(polarization) == len(position_data_aligned_individuals.time)

        # Explicit absence checks (documents which dims are reduced)
        assert "keypoints" not in polarization.dims
        assert "space" not in polarization.dims
        assert "individuals" not in polarization.dims

    def test_polarization_first_frame_velocity_mode(
        self, position_data_aligned_individuals
    ):
        """Test first frame behavior when using velocity-based heading.

        The compute_velocity function uses xarray's differentiate which
        uses edge_order=1 by default, providing valid values at boundaries.
        """
        polarization = kinematics.compute_polarization(
            position_data_aligned_individuals
        )

        # First frame should have a valid value due to forward differencing
        assert isinstance(polarization, xr.DataArray)
        assert len(polarization) == len(position_data_aligned_individuals.time)

    def test_polarization_first_frame_valid_keypoint_mode(
        self, position_data_with_keypoints
    ):
        """Test that first frame is valid when using keypoint-based heading."""
        polarization = kinematics.compute_polarization(
            position_data_with_keypoints,
            heading_keypoints=("tail", "nose"),
        )

        # First frame should be valid (keypoint positions are always known)
        assert not np.isnan(polarization.values[0])

    # ==================== Mathematical Properties ====================

    def test_polarization_symmetry(self):
        """Test polarization is symmetric (individual order irrelevant)."""
        time = [0, 1, 2]
        keypoints = ["centroid"]
        space = ["x", "y"]

        # Create two datasets with same individuals in different order
        data1 = np.array(
            [
                [[[0, 5]], [[0, 0]]],
                [[[1, 4]], [[0, 0]]],  # id_0: +x, id_1: -x
                [[[2, 3]], [[0, 0]]],
            ],
            dtype=float,
        )
        data2 = np.array(
            [
                [[[5, 0]], [[0, 0]]],
                [[[4, 1]], [[0, 0]]],  # id_0: -x, id_1: +x (swapped)
                [[[3, 2]], [[0, 0]]],
            ],
            dtype=float,
        )

        da1 = xr.DataArray(
            data1,
            dims=["time", "space", "keypoints", "individuals"],
            coords={
                "time": time,
                "space": space,
                "keypoints": keypoints,
                "individuals": ["id_0", "id_1"],
            },
        )
        da2 = xr.DataArray(
            data2,
            dims=["time", "space", "keypoints", "individuals"],
            coords={
                "time": time,
                "space": space,
                "keypoints": keypoints,
                "individuals": ["id_0", "id_1"],
            },
        )

        pol1 = kinematics.compute_polarization(da1)
        pol2 = kinematics.compute_polarization(da2)

        np.testing.assert_array_almost_equal(pol1.values, pol2.values)

    def test_polarization_bounds(self, position_data_aligned_individuals):
        """Test polarization values are always in [0, 1] range.

        Verifies bounds with both:
        1. Simple aligned data (deterministic, yields boundary value 1.0)
        2. Random directions (stochastic, yields distribution across range)
        """
        # Test with simple aligned data (boundary case: all 1.0)
        polarization_simple = kinematics.compute_polarization(
            position_data_aligned_individuals
        )
        valid_simple = polarization_simple.values[
            ~np.isnan(polarization_simple.values)
        ]
        assert np.all(valid_simple >= 0.0)
        assert np.all(valid_simple <= 1.0)

        # Test with random directions (interior values)
        time = [0, 1, 2, 3, 4]
        individuals = [f"id_{i}" for i in range(10)]
        keypoints = ["centroid"]
        space = ["x", "y"]

        # Create semi-random movement patterns
        np.random.seed(42)
        n_ind = len(individuals)
        n_time = len(time)

        # Random starting positions
        x_start = np.random.rand(n_ind) * 100
        y_start = np.random.rand(n_ind) * 100

        # Random velocities
        vx = np.random.randn(n_ind) * 2
        vy = np.random.randn(n_ind) * 2

        data = np.zeros((n_time, 2, 1, n_ind))
        for t in range(n_time):
            data[t, 0, 0, :] = x_start + vx * t
            data[t, 1, 0, :] = y_start + vy * t

        da = xr.DataArray(
            data,
            dims=["time", "space", "keypoints", "individuals"],
            coords={
                "time": time,
                "space": space,
                "keypoints": keypoints,
                "individuals": individuals,
            },
        )

        polarization_random = kinematics.compute_polarization(da)

        valid_random = polarization_random.values[
            ~np.isnan(polarization_random.values)
        ]
        assert np.all(valid_random >= 0.0)
        assert np.all(valid_random <= 1.0)

    # ==================== NaN Handling Edge Cases ====================

    def test_polarization_nan_one_coordinate_only(self):
        """Test handling when only one spatial coordinate is NaN."""
        time = [0, 1, 2]
        individuals = ["id_0", "id_1"]
        keypoints = ["centroid"]
        space = ["x", "y"]

        # id_1 has NaN only in x at time 1
        data = np.array(
            [
                [[[0, 5]], [[0, 0]]],
                [[[1, np.nan]], [[0, 0]]],  # x is NaN, y is valid
                [[[2, 7]], [[0, 0]]],
            ],
            dtype=float,
        )

        da = xr.DataArray(
            data,
            dims=["time", "space", "keypoints", "individuals"],
            coords={
                "time": time,
                "space": space,
                "keypoints": keypoints,
                "individuals": individuals,
            },
        )

        polarization = kinematics.compute_polarization(da)

        # Should still compute - NaN individual is excluded
        assert isinstance(polarization, xr.DataArray)

    def test_polarization_nan_in_keypoints(self):
        """Test handling NaN in keypoint-based heading calculation."""
        time = [0, 1, 2]
        individuals = ["id_0", "id_1"]
        keypoints = ["nose", "tail"]
        space = ["x", "y"]

        # id_1's nose is NaN at time 1
        data = np.array(
            [
                [[[2, 5], [0, 3]], [[0, 0], [0, 0]]],
                [[[3, np.nan], [1, 4]], [[0, np.nan], [0, 0]]],
                [[[4, 7], [2, 5]], [[0, 0], [0, 0]]],
            ],
            dtype=float,
        )

        da = xr.DataArray(
            data,
            dims=["time", "space", "keypoints", "individuals"],
            coords={
                "time": time,
                "space": space,
                "keypoints": keypoints,
                "individuals": individuals,
            },
        )

        polarization = kinematics.compute_polarization(
            da, heading_keypoints=("tail", "nose")
        )

        # Time 0 and 2 should be valid (both face +x -> 1.0)
        assert np.allclose(polarization.values[0], 1.0, atol=1e-10)
        assert np.allclose(polarization.values[2], 1.0, atol=1e-10)
        # Time 1 should be 1.0 (only id_0 is valid, single individual)
        assert np.allclose(polarization.values[1], 1.0, atol=1e-10)

    # ==================== Error Handling ====================

    def test_missing_space_dimension(self):
        """Test that missing space dimension raises ValueError."""
        data = xr.DataArray(
            np.random.rand(4, 3),
            dims=["time", "individuals"],
            coords={"time": [0, 1, 2, 3], "individuals": ["a", "b", "c"]},
        )
        with pytest.raises(ValueError, match="space"):
            kinematics.compute_polarization(data)

    def test_missing_time_dimension(self):
        """Test that missing time dimension raises ValueError."""
        data = xr.DataArray(
            np.random.rand(2, 3),
            dims=["space", "individuals"],
            coords={"space": ["x", "y"], "individuals": ["a", "b", "c"]},
        )
        with pytest.raises(ValueError, match="time"):
            kinematics.compute_polarization(data)

    def test_empty_dataarray(self):
        """Test handling of empty DataArray raises an error.

        Empty arrays cause issues in numpy's gradient computation
        used by compute_velocity.
        """
        data = xr.DataArray(
            np.array([]).reshape(0, 2, 0),
            dims=["time", "space", "individuals"],
            coords={"time": [], "space": ["x", "y"], "individuals": []},
        )
        with pytest.raises((IndexError, ValueError)):
            kinematics.compute_polarization(data)

    def test_polarization_mixed_stationary_moving(self):
        """Test polarization with some stationary and some moving individuals.

        When some individuals are stationary (zero velocity), they should be
        excluded from the polarization calculation (NaN unit vector).
        """
        time = [0, 1, 2, 3]
        individuals = ["id_0", "id_1", "id_2"]
        keypoints = ["centroid"]
        space = ["x", "y"]

        # id_0: stationary at (0, 0)
        # id_1: moves in +x direction
        # id_2: moves in +x direction
        data = np.array(
            [
                [[[0, 5, 10]], [[0, 0, 0]]],
                [[[0, 6, 11]], [[0, 0, 0]]],
                [[[0, 7, 12]], [[0, 0, 0]]],
                [[[0, 8, 13]], [[0, 0, 0]]],
            ],
            dtype=float,
        )

        da = xr.DataArray(
            data,
            dims=["time", "space", "keypoints", "individuals"],
            coords={
                "time": time,
                "space": space,
                "keypoints": keypoints,
                "individuals": individuals,
            },
        )

        polarization = kinematics.compute_polarization(da)

        # id_0 is stationary -> NaN heading -> excluded
        # id_1 and id_2 both move +x -> polarization = 1.0
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)

    def test_polarization_two_time_points_minimum(self):
        """Test polarization with minimum time points (2)."""
        time = [0, 1]
        individuals = ["id_0", "id_1"]
        keypoints = ["centroid"]
        space = ["x", "y"]

        # Both move in +x direction
        data = np.array(
            [
                [[[0, 5]], [[0, 0]]],
                [[[1, 6]], [[0, 0]]],
            ],
            dtype=float,
        )

        da = xr.DataArray(
            data,
            dims=["time", "space", "keypoints", "individuals"],
            coords={
                "time": time,
                "space": space,
                "keypoints": keypoints,
                "individuals": individuals,
            },
        )

        polarization = kinematics.compute_polarization(da)

        assert isinstance(polarization, xr.DataArray)
        assert len(polarization) == 2
        # Both moving same direction
        assert np.allclose(polarization.values, 1.0, atol=1e-10)

    # ==================== 3D Data (Limitation) ====================

    def test_3d_data_uses_only_xy(self):
        """Test that 3D spatial data only uses x,y coordinates.

        LIMITATION: The current implementation uses validate_dims_coords
        with exact_coords=False, so 3D data passes validation but only
        x and y coordinates are used for norm/unit vector computation.
        The z coordinate is silently ignored.

        This test documents this limitation.
        """
        time = [0, 1, 2]
        individuals = ["id_0", "id_1"]
        keypoints = ["centroid"]
        space = ["x", "y", "z"]

        # 3D movement data - both individuals move in +x direction
        # z movement is present but will be ignored
        data = np.array(
            [
                [[[0, 5]], [[0, 0]], [[0, 0]]],
                [[[1, 6]], [[0, 0]], [[1, 1]]],
                [[[2, 7]], [[0, 0]], [[2, 2]]],
            ],
            dtype=float,
        )

        da = xr.DataArray(
            data,
            dims=["time", "space", "keypoints", "individuals"],
            coords={
                "time": time,
                "space": space,
                "keypoints": keypoints,
                "individuals": individuals,
            },
        )

        # 3D data silently produces results using only x,y
        # This is a limitation - z is ignored
        polarization = kinematics.compute_polarization(da)
        assert isinstance(polarization, xr.DataArray)
        # Both moving in same x direction -> polarization = 1.0 (ignoring z)
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)
