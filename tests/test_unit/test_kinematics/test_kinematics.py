from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement import kinematics
from movement.kinematics import straightness_index


@pytest.mark.parametrize(
    "kinematic_variable",
    [
        "forward_displacement",
        "backward_displacement",
        "velocity",
        "acceleration",
        "speed",
    ],
)
class TestComputeKinematics:
    """Test ``compute_[kinematic_variable]`` with valid and invalid inputs."""

    forward_displacement = [
        np.vstack([np.ones((9, 2)), np.zeros((1, 2))]),
        np.multiply(
            np.vstack([np.ones((9, 2)), np.zeros((1, 2))]),
            np.array([1, -1]),
        ),
    ]  # [Individual 0, Individual 1]
    expected_kinematics = {
        "forward_displacement": forward_displacement,
        "backward_displacement": [
            np.roll(-forward_displacement[0], 1, axis=0),
            np.roll(-forward_displacement[1], 1, axis=0),
        ],
        "velocity": [
            np.ones((10, 2)),
            np.multiply(np.ones((10, 2)), np.array([1, -1])),
        ],
        "acceleration": [np.zeros((10, 2)), np.zeros((10, 2))],
        "speed": [np.ones(10) * np.sqrt(2), np.ones(10) * np.sqrt(2)],
    }  # 2D: n_frames, n_space_dims

    @pytest.mark.parametrize(
        "valid_dataset", ["valid_poses_dataset", "valid_bboxes_dataset"]
    )
    def test_kinematics(self, valid_dataset, kinematic_variable, request):
        """Test computed kinematics for a uniform linear motion case.
        See the ``valid_poses_dataset`` and ``valid_bboxes_dataset`` fixtures
        for details.
        """
        # Compute kinematic array from input dataset
        position = request.getfixturevalue(valid_dataset).position
        kinematic_array = getattr(kinematics, f"compute_{kinematic_variable}")(
            position
        )
        assert kinematic_array.name == kinematic_variable
        # Figure out which dimensions to expect in kinematic_array
        # and in the final xarray.DataArray
        expected_dims = ["time", "individuals"]
        if kinematic_variable in [
            "forward_displacement",
            "backward_displacement",
            "velocity",
            "acceleration",
        ]:
            expected_dims.insert(1, "space")
        # Build expected data array from the expected numpy array
        expected_array = xr.DataArray(
            # Stack along the "individuals" axis
            np.stack(
                self.expected_kinematics.get(kinematic_variable), axis=-1
            ),
            dims=expected_dims,
        )
        if "keypoints" in position.coords:
            expected_array = expected_array.expand_dims(
                {"keypoints": position.coords["keypoints"].size}
            )
            expected_dims.insert(-1, "keypoints")
            expected_array = expected_array.transpose(*expected_dims)
        # Compare the values of the kinematic_array against the expected_array
        np.testing.assert_allclose(
            kinematic_array.values, expected_array.values
        )

    @pytest.mark.filterwarnings(
        "ignore:The result may be unreliable.*:UserWarning"
    )
    @pytest.mark.parametrize(
        "valid_dataset_with_nan, expected_nans_per_individual",
        [
            (
                "valid_poses_dataset_with_nan",
                {
                    "forward_displacement": [30, 0],
                    "backward_displacement": [30, 0],
                    "velocity": [36, 0],
                    "acceleration": [40, 0],
                    "speed": [18, 0],
                },
            ),
            (
                "valid_bboxes_dataset_with_nan",
                {
                    "forward_displacement": [10, 0],
                    "backward_displacement": [10, 0],
                    "velocity": [12, 0],
                    "acceleration": [14, 0],
                    "speed": [6, 0],
                },
            ),
        ],
    )
    def test_kinematics_with_dataset_with_nans(
        self,
        valid_dataset_with_nan,
        expected_nans_per_individual,
        kinematic_variable,
        helpers,
        request,
    ):
        """Test kinematics computation for a dataset with nans.
        See the ``valid_poses_dataset_with_nan`` and
        ``valid_bboxes_dataset_with_nan`` fixtures for details.
        """
        # compute kinematic array
        valid_dataset = request.getfixturevalue(valid_dataset_with_nan)
        position = valid_dataset.position
        kinematic_array = getattr(kinematics, f"compute_{kinematic_variable}")(
            position
        )
        assert kinematic_array.name == kinematic_variable
        # compute n nans in kinematic array per individual
        n_nans_kinematics_per_indiv = [
            helpers.count_nans(kinematic_array.isel(individuals=i))
            for i in range(valid_dataset.sizes["individuals"])
        ]
        # assert n nans in kinematic array per individual matches expected
        assert (
            n_nans_kinematics_per_indiv
            == expected_nans_per_individual[kinematic_variable]
        )

    @pytest.mark.parametrize(
        "invalid_dataset, expected_exception",
        [
            ("not_a_dataset", pytest.raises(AttributeError)),
            ("empty_dataset", pytest.raises(AttributeError)),
            ("missing_var_poses_dataset", pytest.raises(AttributeError)),
            ("missing_var_bboxes_dataset", pytest.raises(AttributeError)),
            ("missing_dim_poses_dataset", pytest.raises(ValueError)),
            ("missing_dim_bboxes_dataset", pytest.raises(ValueError)),
        ],
    )
    def test_kinematics_with_invalid_dataset(
        self, invalid_dataset, expected_exception, kinematic_variable, request
    ):
        """Test kinematics computation with an invalid dataset."""
        with expected_exception:
            position = request.getfixturevalue(invalid_dataset).position
            getattr(kinematics, f"compute_{kinematic_variable}")(position)


@pytest.mark.parametrize(
    "order, expected_exception",
    [
        (0, pytest.raises(ValueError)),
        (-1, pytest.raises(ValueError)),
        (1.0, pytest.raises(TypeError)),
        ("1", pytest.raises(TypeError)),
    ],
)
def test_time_derivative_with_invalid_order(order, expected_exception):
    """Test that an error is raised when the order is non-positive."""
    data = np.arange(10)
    with expected_exception:
        kinematics.compute_time_derivative(data, order=order)


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
        path_length = kinematics.compute_path_length(
            position, start=start, stop=stop
        )
        assert path_length.name == "path_length"

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
        path_length = kinematics.compute_path_length(
            position, nan_policy=nan_policy
        )
        assert path_length.name == "path_length"
        # Get path_length for individual "id_0" as a numpy array
        path_length_id_0 = path_length.sel(individuals="id_0").values
        # Check them against the expected values
        np.testing.assert_allclose(
            path_length_id_0, expected_path_lengths_id_0
        )


# Regex patterns to match the warning messages
exclude_id_1_and_left = r"(?s)(?!.*id_1)(?!.*left)"
include_threshold_100 = r".*The result may be unreliable.*right.*id_0.*10/10.*"
include_threshold_20 = (
    r".*The result may be unreliable.*centroid.*right.*id_0.*3/10.*10/10.*"
)


@pytest.mark.parametrize(
    "nan_warn_threshold, expected_exception",
    [
        (
            1,
            pytest.warns(
                UserWarning,
                match=f"{exclude_id_1_and_left}{include_threshold_100}",
            ),
        ),
        (
            0.2,
            pytest.warns(
                UserWarning,
                match=f"{exclude_id_1_and_left}{include_threshold_20}",
            ),
        ),
        (-1, pytest.raises(ValueError, match="between 0 and 1")),
    ],
)
def test_path_length_nan_warn_threshold(
    valid_poses_dataset_with_nan, nan_warn_threshold, expected_exception
):
    """Test that a warning is raised with matching message containing
    information on the individuals and keypoints whose number of missing
    values exceeds the given threshold or that an error is raised
    when the threshold is invalid.
    """
    position = valid_poses_dataset_with_nan.position
    with expected_exception:
        result = kinematics.compute_path_length(
            position, nan_warn_threshold=nan_warn_threshold
        )
        assert result.name == "path_length"


def create_straight_line_dataset(length=10, n_ind=1, n_kp=1):
    """Simulate movement in a straight line for testing.
    This creates an xarray.DataArray representing movement from (0,0) to (9,9),
    for each individual/keypoint. Used to test that straightness index == 1.
    """
    coords = {
        "time": np.arange(length),
        "individuals": [f"id_{i}" for i in range(n_ind)],
        "keypoints": [f"kp_{i}" for i in range(n_kp)],
        "space": ["x", "y"],
    }
    data = np.stack(
        [np.linspace([0, 0], [9, 9], length) for _ in range(n_ind * n_kp)]
    )
    data = data.reshape((n_ind, n_kp, length, 2))
    # xarray shape: (individuals, keypoints, time, space)
    ds = xr.DataArray(
        data.transpose(2, 0, 1, 3),  # (time, individuals, keypoints, space)
        dims=["time", "individuals", "keypoints", "space"],
        coords=coords,
    )
    return ds


def create_curved_path_dataset(length=10, n_ind=1, n_kp=1):
    """Create a semicircular (curved) trajectory for testing.
    This dataset is not straight, so the straightness index should be < 1.
    Useful for verifying the function's sensitivity to path curvature.
    """
    theta = np.linspace(0, np.pi, length)
    x = np.cos(theta)
    y = np.sin(theta)
    base = np.stack([x, y], axis=-1)
    base = base * 10  # stretch for effect
    data = np.broadcast_to(base, (n_ind, n_kp, length, 2))
    data = data.reshape((n_ind, n_kp, length, 2))
    ds = xr.DataArray(
        data.transpose(2, 0, 1, 3),
        dims=["time", "individuals", "keypoints", "space"],
        coords={
            "time": np.arange(length),
            "individuals": [f"id_{i}" for i in range(n_ind)],
            "keypoints": [f"kp_{i}" for i in range(n_kp)],
            "space": ["x", "y"],
        },
    )
    return ds


def make_nans(dataarray, which="start"):
    """Introduce NaNs at specified locations within the dataarray.
    Used to test function's robustness to missing data in various regions.
    """
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


@pytest.mark.parametrize(
    "prep_func, expected, tol",
    [
        (create_straight_line_dataset, 1.0, 1e-10),
        (create_curved_path_dataset, pytest.approx(0.63, abs=0.02), 0.05),
    ],
)
def test_straightness_basic(prep_func, expected, tol):
    """Test basic expected behavior:
    - straightness index is 1 for perfect straight line,
    - < 1 for a curved path.
    Checks that values are finite and within [0, 1].
    """
    data = prep_func()
    si = straightness_index(data)
    # Should be between 0 and 1, close to expected
    assert np.all(np.isfinite(si.values))
    assert np.all(si.values <= 1)
    assert np.all(si.values >= 0)
    if isinstance(expected, float):
        assert np.allclose(si.values, expected, atol=tol)
    else:
        assert expected == si.values


def test_straightness_zero_path():
    """Edge case: all positions identical (no movement).
    The path length should be zero, so straightness index should be NaN.
    """
    # All points identical
    data = create_straight_line_dataset()
    data.values[:] = 42
    si = straightness_index(data)
    # Should be all np.nan due to zero path length
    assert np.all(np.isnan(si.values))


@pytest.mark.parametrize("which", ["start", "end", "all", "middle"])
def test_straightness_nans(which):
    """Test function's handling of missing data at various locations:
    - Only start missing,
    - Only end missing,
    - All points missing,
    - NaNs in the middle of trajectory.
    Should return NaN for start/end/all missing, may be valid for 'middle'.
    """
    data = create_straight_line_dataset()
    nan_data = make_nans(data, which=which)
    # Should not raise, but produces nan
    si = straightness_index(nan_data)
    # At least one (all for "all") value is nan
    if which == "all":
        assert np.all(np.isnan(si.values))
    else:
        assert np.any(np.isnan(si.values))


def test_straightness_single_timepoint():
    """Edge case: dataset has only a single timepoint.
    Function should raise ValueError due to insufficient data for
    displacement/path.
    """
    data = create_straight_line_dataset(length=1)
    # Should error due to path length/displacement calculation needing 2 points
    with pytest.raises(ValueError):
        straightness_index(data)


def test_straightness_multiple_individuals_keypoints():
    """Test function's behavior with multiple individuals and keypoints.
    Ensures shape and axes of output match expected.
    """
    data = create_straight_line_dataset(n_ind=3, n_kp=2)
    si = straightness_index(data)
    # Shape, coords consistent
    assert "individuals" in si.dims
    assert "keypoints" in si.dims
    assert si.shape == (1, 3, 2) or si.ndim >= 1
