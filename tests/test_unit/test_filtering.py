from contextlib import nullcontext as does_not_raise

import pytest
import xarray as xr

from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    kalman_filter,
    rolling_filter,
    savgol_filter,
)

# Dataset fixtures
list_valid_datasets_without_nans = [
    "valid_poses_dataset",
    "valid_bboxes_dataset",
]
list_valid_datasets_with_nans = [
    f"{dataset}_with_nan" for dataset in list_valid_datasets_without_nans
]
list_all_valid_datasets = (
    list_valid_datasets_without_nans + list_valid_datasets_with_nans
)


@pytest.mark.parametrize(
    "valid_dataset",
    list_all_valid_datasets,
)
class TestFilteringValidDataset:
    """Test rolling and savgol filtering on
    valid datasets with/without NaNs.
    """

    @pytest.mark.parametrize(
        ("filter_func, filter_kwargs"),
        [
            (rolling_filter, {"window": 3, "statistic": "median"}),
            (savgol_filter, {"window": 3, "polyorder": 2}),
        ],
    )
    def test_filter_with_nans_on_position(
        self, filter_func, filter_kwargs, valid_dataset, helpers, request
    ):
        """Test NaN behaviour of the rolling and SG filters.
        Both filters should set all values to NaN if one element of the
        sliding window is NaN.
        """
        # Expected number of nans in the position array per individual
        expected_nans_in_filtered_position_per_indiv = {
            "valid_poses_dataset": [0, 0],  # no nans in input
            "valid_bboxes_dataset": [0, 0],  # no nans in input
            "valid_poses_dataset_with_nan": [38, 0],
            "valid_bboxes_dataset_with_nan": [14, 0],
        }
        # Filter position
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        position_filtered = filter_func(
            valid_input_dataset.position, **filter_kwargs, print_report=True
        )
        # Compute n nans in position after filtering per individual
        n_nans_after_filtering_per_indiv = [
            helpers.count_nans(position_filtered.isel(individuals=i))
            for i in range(valid_input_dataset.sizes["individuals"])
        ]
        # Check number of nans per indiv is as expected
        assert (
            n_nans_after_filtering_per_indiv
            == expected_nans_in_filtered_position_per_indiv[valid_dataset]
        )

    @pytest.mark.parametrize(
        "override_kwargs, expected_exception",
        [
            ({"mode": "nearest", "print_report": True}, does_not_raise()),
            ({"axis": 1}, pytest.raises(ValueError)),
            ({"mode": "nearest", "axis": 1}, pytest.raises(ValueError)),
        ],
    )
    def test_savgol_filter_kwargs_override(
        self, valid_dataset, override_kwargs, expected_exception, request
    ):
        """Test that overriding keyword arguments in the
        Savitzky-Golay filter works, except for the ``axis`` argument,
        which should raise a ValueError.
        """
        with expected_exception:
            savgol_filter(
                request.getfixturevalue(valid_dataset).position,
                window=3,
                **override_kwargs,
            )

    @pytest.mark.parametrize(
        "statistic, expected_exception",
        [
            ("mean", does_not_raise()),
            ("median", does_not_raise()),
            ("max", does_not_raise()),
            ("min", does_not_raise()),
            ("invalid", pytest.raises(ValueError, match="Invalid statistic")),
        ],
    )
    def test_rolling_filter_statistic(
        self, valid_dataset, statistic, expected_exception, request
    ):
        """Test that the rolling filter works with different statistics."""
        with expected_exception:
            rolling_filter(
                request.getfixturevalue(valid_dataset).position,
                window=3,
                statistic=statistic,
            )


@pytest.mark.parametrize(
    "valid_dataset_with_nan",
    list_valid_datasets_with_nans,
)
class TestFilteringValidDatasetWithNaNs:
    """Test filtering functions on datasets with NaNs."""

    @pytest.mark.parametrize(
        "max_gap, expected_n_nans_in_position",
        [(None, [22, 0]), (0, [28, 6]), (1, [26, 4]), (2, [22, 0])],
        # expected total n nans: [poses, bboxes]
    )
    def test_interpolate_over_time_on_position(
        self,
        valid_dataset_with_nan,
        max_gap,
        expected_n_nans_in_position,
        helpers,
        request,
    ):
        """Test that the number of NaNs decreases after linearly interpolating
        over time and that the resulting number of NaNs is as expected
        for different values of ``max_gap``.
        """
        valid_dataset_in_frames = request.getfixturevalue(
            valid_dataset_with_nan
        )
        # Get position array with time unit in frames & seconds
        # assuming 10 fps = 0.1 s per frame
        valid_dataset_in_seconds = valid_dataset_in_frames.copy()
        valid_dataset_in_seconds.coords["time"] = (
            valid_dataset_in_seconds.coords["time"] * 0.1
        )
        position = {
            "frames": valid_dataset_in_frames.position,
            "seconds": valid_dataset_in_seconds.position,
        }
        # Count number of NaNs
        n_nans_after_per_time_unit = {}
        for time_unit in ["frames", "seconds"]:
            # interpolate
            position_interp = interpolate_over_time(
                position[time_unit],
                method="linear",
                max_gap=max_gap,
                print_report=True,
            )
            # count nans
            n_nans_after_per_time_unit[time_unit] = helpers.count_nans(
                position_interp
            )
        # The number of NaNs should be the same for both datasets
        # as max_gap is based on number of missing observations (NaNs)
        assert (
            n_nans_after_per_time_unit["frames"]
            == n_nans_after_per_time_unit["seconds"]
        )
        # The number of NaNs after interpolating should be as expected
        n_nans_after = n_nans_after_per_time_unit["frames"]
        dataset_index = list_valid_datasets_with_nans.index(
            valid_dataset_with_nan
        )
        assert n_nans_after == expected_n_nans_in_position[dataset_index]

    @pytest.mark.parametrize(
        "window",
        [3, 5, 6, 10],  # input data has 10 frames
    )
    @pytest.mark.parametrize("filter_func", [rolling_filter, savgol_filter])
    def test_filter_with_nans_on_position_varying_window(
        self, valid_dataset_with_nan, window, filter_func, helpers, request
    ):
        """Test that the number of NaNs in the filtered position data
        increases at most by the filter's window length minus one
        multiplied by the number of consecutive NaNs in the input data.
        """
        # Prepare kwargs per filter
        kwargs = {"window": window}
        if filter_func == savgol_filter:
            kwargs["polyorder"] = 2
        # Filter position
        valid_input_dataset = request.getfixturevalue(valid_dataset_with_nan)
        position_filtered = filter_func(
            valid_input_dataset.position,
            **kwargs,
        )
        # Count number of NaNs in the input and filtered position data
        n_total_nans_initial = helpers.count_nans(valid_input_dataset.position)
        n_consecutive_nans_initial = helpers.count_consecutive_nans(
            valid_input_dataset.position
        )
        n_total_nans_filtered = helpers.count_nans(position_filtered)
        max_nans_increase = (window - 1) * n_consecutive_nans_initial
        # Check that filtering does not reduce number of nans
        assert n_total_nans_filtered >= n_total_nans_initial
        # Check that the increase in nans is below the expected threshold
        assert (
            n_total_nans_filtered - n_total_nans_initial <= max_nans_increase
        )


@pytest.mark.parametrize(
    "valid_dataset_no_nans",
    list_valid_datasets_without_nans,
)
def test_filter_by_confidence_on_position(
    valid_dataset_no_nans, helpers, request
):
    """Test that points below the default 0.6 confidence threshold
    are converted to NaN.
    """
    # Filter position by confidence
    valid_input_dataset = request.getfixturevalue(valid_dataset_no_nans)
    position_filtered = filter_by_confidence(
        valid_input_dataset.position,
        confidence=valid_input_dataset.confidence,
        threshold=0.6,
        print_report=True,
    )
    # Count number of NaNs in the full array
    n_nans = helpers.count_nans(position_filtered)
    # expected number of nans for poses:
    # 5 timepoints * 2 individuals * 2 keypoints
    # Note: we count the number of nans in the array, so we multiply
    # the number of low confidence keypoints by the number of
    # space dimensions
    n_low_confidence_kpts = 5
    assert isinstance(position_filtered, xr.DataArray)
    assert n_nans == valid_input_dataset.sizes["space"] * n_low_confidence_kpts


@pytest.mark.parametrize(
    "valid_dataset",
    list_all_valid_datasets,
)
class TestKalmanFilter:
    """Test Kalman filter on valid datasets with/without NaNs."""

    def test_kalman_filter_position_output(
        self, valid_dataset, helpers, request
    ):
        """Test that Kalman filter returns position with correct shape."""
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        position_filtered = kalman_filter(
            valid_input_dataset.position,
            process_noise=0.01,
            measurement_noise=1.0,
            output="position",
        )
        assert isinstance(position_filtered, xr.DataArray)
        assert position_filtered.dims == valid_input_dataset.position.dims
        assert position_filtered.sizes == valid_input_dataset.position.sizes
        assert position_filtered.name == "position"

    def test_kalman_filter_velocity_output(
        self, valid_dataset, helpers, request
    ):
        """Test that Kalman filter returns velocity with correct shape."""
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        velocity = kalman_filter(
            valid_input_dataset.position,
            process_noise=0.01,
            measurement_noise=1.0,
            output="velocity",
        )
        assert isinstance(velocity, xr.DataArray)
        assert velocity.dims == valid_input_dataset.position.dims
        assert velocity.sizes == valid_input_dataset.position.sizes
        assert velocity.name == "velocity"

    def test_kalman_filter_acceleration_output(
        self, valid_dataset, helpers, request
    ):
        """Test that Kalman filter returns acceleration with correct shape."""
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        acceleration = kalman_filter(
            valid_input_dataset.position,
            process_noise=0.01,
            measurement_noise=1.0,
            output="acceleration",
        )
        assert isinstance(acceleration, xr.DataArray)
        assert acceleration.dims == valid_input_dataset.position.dims
        assert acceleration.sizes == valid_input_dataset.position.sizes
        assert acceleration.name == "acceleration"

    def test_kalman_filter_all_output(self, valid_dataset, helpers, request):
        """Test that Kalman filter returns all outputs as Dataset."""
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        results = kalman_filter(
            valid_input_dataset.position,
            process_noise=0.01,
            measurement_noise=1.0,
            output="all",
        )
        assert isinstance(results, xr.Dataset)
        assert "position" in results.data_vars
        assert "velocity" in results.data_vars
        assert "acceleration" in results.data_vars
        assert results.position.dims == valid_input_dataset.position.dims
        assert results.velocity.dims == valid_input_dataset.position.dims
        assert results.acceleration.dims == valid_input_dataset.position.dims

    def test_kalman_filter_with_dt(self, valid_dataset, helpers, request):
        """Test that Kalman filter works with explicit dt."""
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        position_filtered = kalman_filter(
            valid_input_dataset.position,
            dt=0.1,
            process_noise=0.01,
            measurement_noise=1.0,
        )
        assert isinstance(position_filtered, xr.DataArray)

    def test_kalman_filter_with_fps(self, valid_dataset, helpers, request):
        """Test that Kalman filter infers dt from fps attribute."""
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        # Add fps attribute
        valid_input_dataset.position.attrs["fps"] = 30.0
        position_filtered = kalman_filter(
            valid_input_dataset.position,
            process_noise=0.01,
            measurement_noise=1.0,
        )
        assert isinstance(position_filtered, xr.DataArray)

    def test_kalman_filter_handles_nans(
        self, valid_dataset, helpers, request
    ):
        """Test that Kalman filter handles NaN values correctly."""
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        position_filtered = kalman_filter(
            valid_input_dataset.position,
            process_noise=0.01,
            measurement_noise=1.0,
            print_report=True,
        )
        # Kalman filter should bridge small gaps, so NaNs may decrease
        # But we don't enforce a specific behavior, just that it doesn't crash
        assert isinstance(position_filtered, xr.DataArray)

    @pytest.mark.parametrize(
        "output, expected_type",
        [
            ("position", xr.DataArray),
            ("velocity", xr.DataArray),
            ("acceleration", xr.DataArray),
            ("all", xr.Dataset),
        ],
    )
    def test_kalman_filter_output_types(
        self, valid_dataset, output, expected_type, request
    ):
        """Test that Kalman filter returns correct output types."""
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        result = kalman_filter(
            valid_input_dataset.position,
            process_noise=0.01,
            measurement_noise=1.0,
            output=output,
        )
        assert isinstance(result, expected_type)

    def test_kalman_filter_invalid_output(self, valid_dataset, request):
        """Test that Kalman filter raises error for invalid output."""
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        with pytest.raises(ValueError, match="Invalid output"):
            kalman_filter(
                valid_input_dataset.position,
                output="invalid",
            )


def test_kalman_filter_invalid_space_dims():
    """Test that Kalman filter raises error for unsupported space dims."""
    # Create a dataset with 1D space (not supported)
    position_1d = xr.DataArray(
        [[0.0], [1.0], [2.0]],
        dims=["time", "space"],
        coords={"time": [0, 1, 2], "space": ["x"]},
    )
    with pytest.raises(ValueError, match="2D or 3D space"):
        kalman_filter(position_1d)
