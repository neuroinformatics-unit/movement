from contextlib import nullcontext as does_not_raise

import pytest
import xarray as xr

from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    median_filter,
    savgol_filter,
)


@pytest.mark.parametrize(
    "valid_dataset_with_nan",
    ["valid_poses_dataset_with_nan", "valid_bboxes_dataset_with_nan"],
)
@pytest.mark.parametrize(
    "max_gap, expected_n_nans_in_position", [(None, 0), (0, 3), (1, 2), (2, 0)]
)
def test_interpolate_over_time(
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
    # First dataset with time unit in frames
    valid_dataset = request.getfixturevalue(valid_dataset_with_nan)
    position_in_frames = valid_dataset.position

    # Create second dataset with time unit in seconds
    position_in_seconds = position_in_frames.copy()
    position_in_seconds["time"] = position_in_seconds["time"] * 0.1

    # Interpolate nans, for data in frames and data in seconds over time
    position_interp_frames = interpolate_over_time(
        position_in_frames, method="linear", max_gap=max_gap
    )
    position_interp_seconds = interpolate_over_time(
        position_in_seconds, method="linear", max_gap=max_gap
    )

    # Count number of NaNs before and after interpolation
    n_nans_before = helpers.count_nans(position_in_frames)
    n_nans_after_frames = helpers.count_nans(position_interp_frames)
    n_nans_after_seconds = helpers.count_nans(position_interp_seconds)

    # The number of NaNs should be the same for both datasets
    # as max_gap is based on number of missing observations (NaNs)
    assert n_nans_after_frames == n_nans_after_seconds

    # The number of NaNs should decrease after interpolation
    if max_gap == 0:
        assert n_nans_after_frames == n_nans_before
    else:
        assert n_nans_after_frames < n_nans_before

    # The number of NaNs after interpolating should be as expected
    assert (
        n_nans_after_frames
        == valid_dataset.dims["space"]
        * valid_dataset.dims.get(
            "keypoints", 1
        )  # in bboxes dataset there is no keypoints dimension
        * expected_n_nans_in_position
    )


@pytest.mark.parametrize(
    "valid_dataset, n_low_confidence_kpts",
    [
        ("valid_poses_dataset", 20),
        ("valid_bboxes_dataset", 5),
    ],
)
def test_filter_by_confidence(
    valid_dataset, n_low_confidence_kpts, helpers, request
):
    """Test that points below the default 0.6 confidence threshold
    are converted to NaN.
    """
    valid_input_dataset = request.getfixturevalue(valid_dataset)

    position = valid_input_dataset.position
    confidence = valid_input_dataset.confidence

    # Filter position by confidence
    position_filtered = filter_by_confidence(
        position, confidence, threshold=0.6
    )

    # Count number of NaNs in the full array
    n_nans = helpers.count_nans(position_filtered)

    # expected number of nans for poses:
    # 5 timepoints * 2 individuals * 2 keypoints
    # Note: we count the number of nans in the array, so we multiply
    # the number of low confidence keypoints by the number of
    # space dimensions
    assert isinstance(position_filtered, xr.DataArray)
    assert n_nans == valid_input_dataset.dims["space"] * n_low_confidence_kpts


@pytest.mark.parametrize(
    "valid_dataset_with_nan",
    ["valid_poses_dataset_with_nan", "valid_bboxes_dataset_with_nan"],
)
@pytest.mark.parametrize("window_size", [2, 4])
def test_median_filter(valid_dataset_with_nan, window_size, request):
    """Test that applying the median filter returns
    a different xr.DataArray than the input data.
    """
    valid_input_dataset = request.getfixturevalue(valid_dataset_with_nan)

    position = valid_input_dataset.position
    position_filtered = median_filter(position, window_size)

    del position_filtered.attrs["log"]

    # filtered array is an xr.DataArray
    assert isinstance(position_filtered, xr.DataArray)

    # filtered data should not be equal to the original data
    assert not position_filtered.equals(position)


def test_median_filter_with_nans(valid_poses_dataset_with_nan, helpers):
    """Test NaN behaviour of the median filter. The input data
    contains NaNs in all keypoints of the first individual at timepoints
    3, 7, and 8 (0-indexed, 10 total timepoints). The median filter
    should propagate NaNs within the windows of the filter,
    but it should not introduce any NaNs for the second individual.
    """
    data = valid_poses_dataset_with_nan.position
    data_smoothed = median_filter(data, window=3)
    # All points of the first individual are converted to NaNs except
    # at timepoints 0, 1, and 5.
    assert not (
        data_smoothed.isel(individuals=0, time=[0, 1, 5]).isnull().any()
    )
    # 7 timepoints * 1 individual * 2 keypoints * 2 space dimensions
    assert helpers.count_nans(data_smoothed) == 28
    # No NaNs should be introduced for the second individual
    assert not data_smoothed.isel(individuals=1).isnull().any()


@pytest.mark.parametrize("window, polyorder", [(2, 1), (4, 2)])
def test_savgol_filter(valid_poses_dataset_with_nan, window, polyorder):
    """Test that applying the Savitzky-Golay filter returns
    a different xr.DataArray than the input data.
    """
    data = valid_poses_dataset_with_nan.position
    data_smoothed = savgol_filter(data, window, polyorder=polyorder)
    del data_smoothed.attrs["log"]
    assert isinstance(data_smoothed, xr.DataArray) and not (
        data_smoothed.equals(data)
    )


def test_savgol_filter_with_nans(valid_poses_dataset_with_nan, helpers):
    """Test NaN behaviour of the Savitzky-Golay filter. The input data
    contains NaN values in all keypoints of the first individual at times
    3, 7, and 8 (0-indexed, 10 total timepoints).
    The Savitzky-Golay filter should propagate NaNs within the windows of
    the filter, but it should not introduce any NaNs for the second individual.
    """
    data = valid_poses_dataset_with_nan.position
    data_smoothed = savgol_filter(data, window=3, polyorder=2)
    # There should be 28 NaNs in total for the first individual, i.e.
    # at 7 timepoints, 2 keypoints, 2 space dimensions
    # all except for timepoints 0, 1 and 5
    assert helpers.count_nans(data_smoothed) == 28
    assert not (
        data_smoothed.isel(individuals=0, time=[0, 1, 5]).isnull().any()
    )
    assert not data_smoothed.isel(individuals=1).isnull().any()


@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"mode": "nearest"},
        {"axis": 1},
        {"mode": "nearest", "axis": 1},
    ],
)
def test_savgol_filter_kwargs_override(
    valid_poses_dataset_with_nan, override_kwargs
):
    """Test that overriding keyword arguments in the Savitzky-Golay filter
    works, except for the ``axis`` argument, which should raise a ValueError.
    """
    expected_exception = (
        pytest.raises(ValueError)
        if "axis" in override_kwargs
        else does_not_raise()
    )
    with expected_exception:
        savgol_filter(
            valid_poses_dataset_with_nan.position,
            window=3,
            **override_kwargs,
        )
