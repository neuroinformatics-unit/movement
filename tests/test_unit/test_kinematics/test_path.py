from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.kinematics import (
    compute_path_length,
    compute_path_straightness,
    compute_roaming_entropy,
)

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def valid_poses_dataset_with_cross_nan(valid_poses_dataset):
    """Return a valid poses dataset with NaNs crossing both dimensions.

    NaN layout (10 frames):
    - (right, id_0): 10/10 NaN (100%)
    - (centroid, id_1): 6/10 NaN (60%)
    - (right, id_1): 1/10 NaN (10%)
    - (centroid, id_0): 1/10 NaN (10%)
    - All other tracks: 0 NaN
    """
    position = valid_poses_dataset.position
    position.loc[{"individual": "id_0", "keypoint": "right"}] = np.nan
    position.loc[
        {
            "individual": "id_1",
            "keypoint": "centroid",
            "time": [0, 1, 2, 3, 4, 5],
        }
    ] = np.nan
    position.loc[{"individual": "id_1", "keypoint": "right", "time": 0}] = (
        np.nan
    )
    position.loc[{"individual": "id_0", "keypoint": "centroid", "time": 0}] = (
        np.nan
    )
    return valid_poses_dataset


@pytest.fixture
def straight_paths(valid_poses_dataset):
    """Return centroid position data from the valid poses dataset.

    Both individuals move in straight lines at constant steps of sqrt(2).
     "id_0" moves along the x=y line, while "id_1" moves along the x=-y line.
    """
    return valid_poses_dataset.position.sel(keypoint="centroid")


@pytest.fixture
def stationary_paths(straight_paths):
    """Return path data where the individuals never move."""
    path = straight_paths.copy()
    path[:] = 3.0
    return path


@pytest.fixture
def closed_loop_paths(straight_paths):
    """Return path data forming a closed loop.

    Overwrites ``straight_paths`` to end at starting point.
    """
    path = straight_paths.copy()
    path.loc[{"time": 9}] = path.loc[{"time": 0}]
    return path


@pytest.fixture
def sharp_turn_paths(straight_paths):
    """Return path data where both individuals take a sharp turn.

    Modifies the ``straight_paths`` fixture.
    After the first 9 frames (8 steps) of straight motion, both individuals
    drop straight back to the x-axis (y=0) for the 10th frame (last step).

    Path length: 8 segments of sqrt(2) + 1 segment of 8 = 8 * sqrt(2) + 8
    Straight-line distance: 8
    Straightness index: (8 / (8 * sqrt(2) + 8)) = 1 / (1 + sqrt(2))
    """
    path = straight_paths.copy()
    # Maintain x position for last step
    path.loc[{"time": 9, "space": "x"}] = path.sel(time=8, space="x")
    # Drop y position to 0 for last step
    path.loc[{"time": 9, "space": "y"}] = 0.0
    return path


# ─────────────────────────────────────────────
# Path length tests
# ─────────────────────────────────────────────

time_points_value_error = pytest.raises(
    ValueError,
    match="At least 2 time points are required to compute path length",
)


@pytest.mark.parametrize(
    "time_slice, expected_exception",
    [
        # full time ranges
        pytest.param(slice(None, None), does_not_raise(), id="full-range"),
        pytest.param(slice(0, None), does_not_raise(), id="from-zero"),
        pytest.param(slice(0, 9), does_not_raise(), id="explicit-full-range"),
        pytest.param(slice(0, 10), does_not_raise(), id="stop-beyond-data"),
        pytest.param(slice(-1, 9), does_not_raise(), id="start-before-data"),
        # partial time ranges
        pytest.param(slice(1, 8), does_not_raise(), id="partial-range"),
        pytest.param(
            slice(1.5, 8.5), does_not_raise(), id="fractional-bounds"
        ),
        pytest.param(slice(2, None), does_not_raise(), id="from-two-to-end"),
        # Empty or too-short slices
        pytest.param(
            slice(9, 0),
            time_points_value_error,
            id="start-greater-than-stop",
        ),
        pytest.param(
            slice(0, 0.5),
            time_points_value_error,
            id="too-few-time-points",
        ),
    ],
)
def test_path_length_across_time_ranges(
    valid_poses_dataset, time_slice, expected_exception
):
    """Test path length computation for a uniform linear motion case,
    across different pre-sliced time ranges.

    The test dataset ``valid_poses_dataset``
    contains 2 individuals ("id_0" and "id_1"), moving
    along x=y and x=-y lines, respectively, at a constant velocity.
    At each frame they cover a distance of sqrt(2) in x-y space, so in total
    we expect a path length of sqrt(2) * num_segments, where num_segments is
    the number of selected frames minus 1.
    """
    position = valid_poses_dataset.position.sel(time=time_slice)
    with expected_exception:
        path_length = compute_path_length(position)
        assert path_length.name == "path_length"
        assert path_length.long_name == "Path Length"

        num_segments = position.sizes["time"] - 1
        expected_path_length = xr.DataArray(
            np.ones((3, 2)) * np.sqrt(2) * num_segments,
            dims=["keypoint", "individual"],
            coords={
                "keypoint": position.coords["keypoint"],
                "individual": position.coords["individual"],
            },
        )
        xr.testing.assert_allclose(path_length, expected_path_length)


@pytest.mark.parametrize(
    "nan_policy, expected_path_lengths_id_0, expected_exception",
    [
        (
            "ffill",
            np.array([np.sqrt(2) * 9, np.sqrt(2) * 8, np.nan]),
            does_not_raise(),
        ),
        (
            "scale",
            np.array([np.sqrt(2) * 9, np.sqrt(2) * 9, np.nan]),
            does_not_raise(),
        ),
        (
            "invalid",  # invalid value for nan_policy
            np.zeros(3),
            pytest.raises(ValueError, match="Invalid value for nan_policy"),
        ),
    ],
)
def test_path_length_with_nan(
    valid_poses_dataset_with_nan,
    nan_policy,
    expected_path_lengths_id_0,
    expected_exception,
):
    """Test path length computation for a uniform linear motion case,
    with varying number of missing values per individual and keypoint.
    Because the underlying motion is uniform linear, the "scale" policy should
    perfectly restore the path length for individual "id_0" to its true value.
    The "ffill" policy should do likewise if frames are missing in the middle,
    but will not "correct" for missing values at the edges.
    """
    position = valid_poses_dataset_with_nan.position
    with (
        pytest.warns(UserWarning, match="The result may be unreliable"),
        expected_exception,
    ):
        path_length = compute_path_length(position, nan_policy=nan_policy)
        assert path_length.name == "path_length"
        # Get path_length for individual "id_0" as a numpy array
        path_length_id_0 = path_length.sel(individual="id_0").values
        # Check them against the expected values
        np.testing.assert_allclose(
            path_length_id_0, expected_path_lengths_id_0
        )


# Regex patterns to match the warning messages
_exclude_id_1_and_left = r"(?s)(?!.*id_1)(?!.*left)"
_include_threshold_100 = (
    r".*The result may be unreliable.*right.*id_0.*10/10.*"
)
_include_threshold_20 = (
    r".*The result may be unreliable"
    r".*centroid.*right.*id_0.*3/10.*10/10.*"
)
_cross_nan_threshold_50 = (
    r"(?s)(?!.*1/10)(?=.*10/10)(?=.*6/10)"
    r".*The result may be unreliable.*"
)


@pytest.mark.parametrize(
    "fixture_name, nan_warn_threshold, expected_exception",
    [
        pytest.param(
            "valid_poses_dataset_with_nan",
            1,
            pytest.warns(
                UserWarning,
                match=(f"{_exclude_id_1_and_left}{_include_threshold_100}"),
            ),
            id="standard-threshold-100",
        ),
        pytest.param(
            "valid_poses_dataset_with_nan",
            0.2,
            pytest.warns(
                UserWarning,
                match=(f"{_exclude_id_1_and_left}{_include_threshold_20}"),
            ),
            id="standard-threshold-20",
        ),
        pytest.param(
            "valid_poses_dataset_with_nan",
            -1,
            pytest.raises(ValueError, match="between 0 and 1"),
            id="invalid-threshold",
        ),
        pytest.param(
            "valid_poses_dataset_with_cross_nan",
            0.5,
            pytest.warns(
                UserWarning,
                match=_cross_nan_threshold_50,
            ),
            id="cross-nan-threshold-50",
        ),
    ],
)
def test_path_length_nan_warn_threshold(
    request,
    fixture_name,
    nan_warn_threshold,
    expected_exception,
):
    """Test that a warning is raised with matching message containing
    information on the individuals and keypoints whose number of missing
    values exceeds the given threshold or that an error is raised
    when the threshold is invalid.
    """
    dataset = request.getfixturevalue(fixture_name)
    position = dataset.position
    with expected_exception:
        result = compute_path_length(
            position, nan_warn_threshold=nan_warn_threshold
        )
        assert result.name == "path_length"


# ─────────────────────────────────────────────
# Path straightness tests
# ─────────────────────────────────────────────

time_points_value_error_straightness = pytest.raises(
    ValueError,
    match=("At least 2 time points are required to compute path straightness"),
)


@pytest.mark.parametrize(
    "time_slice, expected_exception",
    [
        pytest.param(
            slice(None, None),
            does_not_raise(),
            id="full-range",
        ),
        pytest.param(
            slice(0, 9),
            does_not_raise(),
            id="explicit-full-range",
        ),
        pytest.param(
            slice(1, 8),
            does_not_raise(),
            id="partial-range",
        ),
        pytest.param(
            slice(9, 0),
            time_points_value_error_straightness,
            id="start-greater-than-stop",
        ),
        pytest.param(
            slice(0, 0.5),
            time_points_value_error_straightness,
            id="too-few-time-points",
        ),
    ],
)
def test_path_straightness_across_time_ranges(
    valid_poses_dataset, time_slice, expected_exception
):
    """Test straightness index for a uniform linear motion case.

    The ``valid_poses_dataset`` contains 2 individuals moving in
    straight lines, so the straightness index should always be 1.0.
    """
    position = valid_poses_dataset.position.sel(time=time_slice)
    with expected_exception:
        result = compute_path_straightness(position)
        assert result.name == "straightness_index"
        assert result.attrs["long_name"] == "Path Straightness Index"
        xr.testing.assert_allclose(
            result,
            xr.ones_like(result),
        )


@pytest.mark.parametrize(
    "fixture_name, expected_value",
    [
        pytest.param(
            "straight_paths",
            1.0,
            id="straight-line",
        ),
        pytest.param(
            "closed_loop_paths",
            0.0,
            id="closed-loop",
        ),
        pytest.param(
            "sharp_turn_paths",
            1 / (1 + np.sqrt(2)),
            id="sharp-turn",
        ),
        pytest.param(
            "stationary_paths",
            np.nan,
            id="stationary",
        ),
    ],
)
@pytest.mark.parametrize("nan_policy", ["ffill", "scale"])
def test_path_straightness_known_values(
    request, fixture_name, expected_value, nan_policy
):
    """Test that the straightness index matches expected values
    for trajectories with known geometry, regardless of NaN handling policy.
    """
    position = request.getfixturevalue(fixture_name)
    result = compute_path_straightness(position, nan_policy=nan_policy)
    if np.isnan(expected_value):
        assert result.isnull().all()
    else:
        xr.testing.assert_allclose(
            result,
            xr.full_like(result, expected_value),
        )


# ─────────────────────────────────────────────
# Roaming entropy tests
# ─────────────────────────────────────────────


@pytest.fixture
def two_individual_paths():
    """Return a (time, space, individual) position array with two individuals.

    ``id_uniform`` visits the four corners of the unit square equally (two
    visits per corner over 8 frames), so on a 2x2 grid it occupies all four
    bins uniformly. ``id_confined`` stays within a single bin for the whole
    trajectory. This makes the roaming entropy analytically known:
    ``id_uniform`` reaches the maximum (ln 4, or 1.0 normalised), while
    ``id_confined`` is 0.
    """
    corners = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1]],
        dtype=float,
    )
    confined = np.full((8, 2), 0.25)
    # Stack to shape (time, space, individual)
    position = np.stack([corners, confined], axis=-1)
    return xr.DataArray(
        position,
        dims=["time", "space", "individual"],
        coords={
            "time": np.arange(8),
            "space": ["x", "y"],
            "individual": ["id_uniform", "id_confined"],
        },
    )


def test_roaming_entropy_known_values(two_individual_paths):
    """Roaming entropy matches analytically known values for a uniform and a
    confined trajectory, both normalised and unnormalised.
    """
    normalised = compute_roaming_entropy(two_individual_paths, bins=2)
    assert normalised.name == "roaming_entropy"
    assert normalised.attrs["long_name"] == "Roaming Entropy"
    # time and space are reduced; individual is preserved
    assert set(normalised.dims) == {"individual"}
    np.testing.assert_allclose(
        normalised.sel(individual="id_uniform").item(), 1.0
    )
    np.testing.assert_allclose(
        normalised.sel(individual="id_confined").item(), 0.0
    )

    unnormalised = compute_roaming_entropy(
        two_individual_paths, bins=2, normalise=False
    )
    np.testing.assert_allclose(
        unnormalised.sel(individual="id_uniform").item(), np.log(4)
    )
    np.testing.assert_allclose(
        unnormalised.sel(individual="id_confined").item(), 0.0
    )


def test_roaming_entropy_accepts_tuple_bins(two_individual_paths):
    """A ``(nx, ny)`` tuple of bins is accepted and gives the same result as
    the equivalent integer for a square grid.
    """
    from_int = compute_roaming_entropy(two_individual_paths, bins=2)
    from_tuple = compute_roaming_entropy(two_individual_paths, bins=(2, 2))
    xr.testing.assert_allclose(from_int, from_tuple)


def test_roaming_entropy_all_nan_track(two_individual_paths):
    """A track with no valid positions yields NaN, without affecting others."""
    data = two_individual_paths.copy()
    data.loc[{"individual": "id_confined"}] = np.nan
    result = compute_roaming_entropy(data, bins=2)
    assert result.sel(individual="id_confined").isnull().item()
    np.testing.assert_allclose(result.sel(individual="id_uniform").item(), 1.0)


def test_roaming_entropy_all_nan_data(two_individual_paths):
    """If no valid positions exist anywhere, all entropies are NaN."""
    data = xr.full_like(two_individual_paths, np.nan)
    result = compute_roaming_entropy(data, bins=2)
    assert result.isnull().all()


@pytest.mark.parametrize(
    "bins, expected_match",
    [
        pytest.param(0, "bins must be a positive", id="zero"),
        pytest.param(-3, "bins must be a positive", id="negative"),
        pytest.param((2,), "bins must be a positive", id="tuple-wrong-length"),
        pytest.param((2, 0), "bins must be a positive", id="tuple-zero"),
    ],
)
def test_roaming_entropy_invalid_bins(
    two_individual_paths, bins, expected_match
):
    """Invalid ``bins`` values raise a ValueError."""
    with pytest.raises(ValueError, match=expected_match):
        compute_roaming_entropy(two_individual_paths, bins=bins)


def test_roaming_entropy_requires_space_coords(two_individual_paths):
    """Data lacking the required 'x'/'y' space coordinates raises an error."""
    data = two_individual_paths.sel(space=["x"])
    with pytest.raises(ValueError, match="space"):
        compute_roaming_entropy(data)
