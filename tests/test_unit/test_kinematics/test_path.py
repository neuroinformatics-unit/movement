from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.kinematics import compute_path_length, compute_path_straightness


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
    position.loc[
        {"individuals": "id_0", "keypoints": "right"}
    ] = np.nan
    position.loc[
        {
            "individuals": "id_1",
            "keypoints": "centroid",
            "time": [0, 1, 2, 3, 4, 5],
        }
    ] = np.nan
    position.loc[
        {"individuals": "id_1", "keypoints": "right", "time": 0}
    ] = np.nan
    position.loc[
        {"individuals": "id_0", "keypoints": "centroid", "time": 0}
    ] = np.nan
    return valid_poses_dataset


time_points_value_error = pytest.raises(
    ValueError,
    match="At least 2 time points are required to compute path length",
)


@pytest.mark.parametrize(
    "start, stop, expected_exception",
    [
        # full time ranges
        (None, None, does_not_raise()),
        (0, None, does_not_raise()),
        (0, 9, does_not_raise()),
        (0, 10, does_not_raise()),  # xarray.sel will truncate to 0, 9
        (-1, 9, does_not_raise()),  # xarray.sel will truncate to 0, 9
        # partial time ranges
        (1, 8, does_not_raise()),
        (1.5, 8.5, does_not_raise()),
        (2, None, does_not_raise()),
        # Empty time ranges
        (9, 0, time_points_value_error),  # start > stop
        ("text", 9, time_points_value_error),  # invalid start type
        # Time range too short
        (0, 0.5, time_points_value_error),
    ],
)
def test_path_length_across_time_ranges(
    valid_poses_dataset, start, stop, expected_exception
):
    """Test path length computation for a uniform linear motion case,
    across different time ranges.

    The test dataset ``valid_poses_dataset``
    contains 2 individuals ("id_0" and "id_1"), moving
    along x=y and x=-y lines, respectively, at a constant velocity.
    At each frame they cover a distance of sqrt(2) in x-y space, so in total
    we expect a path length of sqrt(2) * num_segments, where num_segments is
    the number of selected frames minus 1.
    """
    position = valid_poses_dataset.position
    with expected_exception:
        path_length = compute_path_length(position, start=start, stop=stop)
        assert path_length.name == "path_length"
        assert path_length.long_name == "Path Length"

        # Expected number of segments (displacements) in selected time range
        num_segments = 9  # full time range: 10 frames - 1
        start = max(0, start) if start is not None else 0
        stop = min(9, stop) if stop is not None else 9
        if start is not None:
            num_segments -= np.ceil(max(0, start))
        if stop is not None:
            stop = min(9, stop)
            num_segments -= 9 - np.floor(min(9, stop))

        expected_path_length = xr.DataArray(
            np.ones((3, 2)) * np.sqrt(2) * num_segments,
            dims=["keypoints", "individuals"],
            coords={
                "keypoints": position.coords["keypoints"],
                "individuals": position.coords["individuals"],
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
        path_length_id_0 = path_length.sel(individuals="id_0").values
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
                match=(
                    f"{_exclude_id_1_and_left}"
                    f"{_include_threshold_100}"
                ),
            ),
            id="standard-threshold-100",
        ),
        pytest.param(
            "valid_poses_dataset_with_nan",
            0.2,
            pytest.warns(
                UserWarning,
                match=(
                    f"{_exclude_id_1_and_left}"
                    f"{_include_threshold_20}"
                ),
            ),
            id="standard-threshold-20",
        ),
        pytest.param(
            "valid_poses_dataset_with_nan",
            -1,
            pytest.raises(
                ValueError, match="between 0 and 1"
            ),
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
    assert result.long_name == "Path Straightness Index"


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
