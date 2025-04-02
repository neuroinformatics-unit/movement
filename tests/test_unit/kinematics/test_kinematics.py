from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.kinematics.kinematics import (
    _compute_scaled_path_length,
    _warn_about_nan_proportion,
    compute_acceleration,
    compute_displacement,
    compute_path_length,
    compute_speed,
    compute_time_derivative,
    compute_velocity,
)


class TestComputeKinematics:
    """Test suite for computing kinematic variables."""

    @pytest.mark.parametrize(
        "valid_dataset", ["valid_poses_dataset", "valid_bboxes_dataset"]
    )
    @pytest.mark.parametrize(
        "kinematic_variable",
        ["displacement", "velocity", "acceleration", "speed", "path_length"],
    )
    def test_kinematics(self, valid_dataset, kinematic_variable, request):
        """Test kinematic computations with valid datasets."""
        position = request.getfixturevalue(valid_dataset).position
        kinematic_func = {
            "displacement": compute_displacement,
            "velocity": compute_velocity,
            "acceleration": compute_acceleration,
            "speed": compute_speed,
            "path_length": compute_path_length,
        }[kinematic_variable]
        kinematic_array = kinematic_func(position)
        assert isinstance(kinematic_array, xr.DataArray)
        if kinematic_variable == "speed":
            expected_dims = tuple(d for d in position.dims if d != "space")
            assert kinematic_array.dims == expected_dims, (
                f"Expected dims {expected_dims}, got {kinematic_array.dims}"
            )
        elif kinematic_variable != "path_length":
            assert "space" in kinematic_array.dims, (
                f"Expected 'space' in dims, got {kinematic_array.dims}"
            )
        else:
            assert (
                "time" not in kinematic_array.dims
                and "space" not in kinematic_array.dims
            ), f"Unexpected dims: {kinematic_array.dims}"

    @pytest.mark.parametrize(
        "valid_dataset_with_nan, expected_nans_per_individual",
        [
            (
                "valid_poses_dataset_with_nan",
                {
                    "displacement": [30, 0],
                    "velocity": [36, 0],
                    "acceleration": [40, 0],
                    "speed": [18, 0],
                    "path_length": [1, 0],
                },
            ),
            (
                "valid_bboxes_dataset_with_nan",
                {
                    "displacement": [10, 0],
                    "velocity": [12, 0],
                    "acceleration": [14, 0],
                    "speed": [6, 0],
                    "path_length": [0, 0],
                },
            ),
        ],
    )
    @pytest.mark.parametrize(
        "kinematic_variable",
        ["displacement", "velocity", "acceleration", "speed", "path_length"],
    )
    def test_kinematics_with_dataset_with_nans(
        self,
        valid_dataset_with_nan,
        expected_nans_per_individual,
        kinematic_variable,
        helpers,
        request,
    ):
        """Test kinematic computations with datasets containing NaNs."""
        valid_dataset = request.getfixturevalue(valid_dataset_with_nan)
        position = valid_dataset.position
        kinematic_func = {
            "displacement": compute_displacement,
            "velocity": compute_velocity,
            "acceleration": compute_acceleration,
            "speed": compute_speed,
            "path_length": compute_path_length,
        }[kinematic_variable]
        kinematic_array = kinematic_func(position)
        expected_nans = expected_nans_per_individual[kinematic_variable]
        actual_nans = np.isnan(kinematic_array).sum().item()
        expected_total_nans = sum(expected_nans)
        assert actual_nans == expected_total_nans, (
            f"{kinematic_variable}: Expected {expected_total_nans} NaNs, "
            f"got {actual_nans}"
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
    @pytest.mark.parametrize(
        "kinematic_variable",
        ["displacement", "velocity", "acceleration", "speed", "path_length"],
    )
    def test_kinematics_with_invalid_dataset(
        self, invalid_dataset, expected_exception, kinematic_variable, request
    ):
        """Test kinematic computations with invalid datasets."""
        with expected_exception:
            position = request.getfixturevalue(invalid_dataset).position
            kinematic_func = {
                "displacement": compute_displacement,
                "velocity": compute_velocity,
                "acceleration": compute_acceleration,
                "speed": compute_speed,
                "path_length": compute_path_length,
            }[kinematic_variable]
            kinematic_func(position)


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
        compute_time_derivative(data, order=order)


# Define time_points_value_error for reuse
time_points_value_error = pytest.raises(
    ValueError,
    match="At least 2 time points are required to compute path length",
)


@pytest.mark.parametrize(
    "nan_warn_threshold, expected_exception",
    [
        (1, does_not_raise()),
        (0.2, does_not_raise()),
        (-1, pytest.raises(ValueError, match="between 0 and 1")),
    ],
)
def test_path_length_warns_about_nans(
    valid_poses_dataset_with_nan,
    nan_warn_threshold,
    expected_exception,
    caplog,
):
    """Test that a warning is raised when the number of missing values
    exceeds a given threshold.
    """
    position = valid_poses_dataset_with_nan.position
    with expected_exception:
        _warn_about_nan_proportion(position, nan_warn_threshold)
        if 0.1 < nan_warn_threshold < 0.5:
            # Make sure that a warning was emitted
            assert caplog.records[0].levelname == "WARNING"
            assert "The result may be unreliable" in caplog.records[0].message
            # Make sure that the NaN report only mentions
            # the individual and keypoint that violate the threshold
            info_msg = caplog.records[1].message
            assert caplog.records[1].levelname == "INFO"
            assert "Individual: id_0" in info_msg
            assert "Individual: id_1" not in info_msg
            assert "centroid: 3/10 (30.0%)" in info_msg
            assert "right: 10/10 (100.0%)" in info_msg
            assert "left" not in info_msg


def test_compute_scaled_path_length(valid_poses_dataset_with_nan):
    """Test scaled path length computation with NaNs."""
    position = valid_poses_dataset_with_nan.position
    path_length = _compute_scaled_path_length(position)
    expected_path_lengths_id_0 = np.array(
        [np.sqrt(2) * 9, np.sqrt(2) * 9, np.nan]
    )
    path_length_id_0 = path_length.sel(individuals="id_0").values
    np.testing.assert_allclose(path_length_id_0, expected_path_lengths_id_0)


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

    The test dataset ``valid_poses_dataset`` contains 2 individuals
    ("id_0" and "id_1"), moving along x=y and x=-y lines, respectively,
    at a constant velocity. At each frame they cover a distance of
    sqrt(2) in x-y space, so in total we expect a path length of
    sqrt(2) * num_segments, where num_segments is the number of
    selected frames minus 1.
    """
    position = valid_poses_dataset.position
    with expected_exception:
        path_length = compute_path_length(position, start=start, stop=stop)
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
            "invalid",
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
    with expected_exception:
        path_length = compute_path_length(position, nan_policy=nan_policy)
        # Get path_length for individual "id_0" as a numpy array
        path_length_id_0 = path_length.sel(individuals="id_0").values
        # Check them against the expected values
        np.testing.assert_allclose(
            path_length_id_0, expected_path_lengths_id_0
        )
