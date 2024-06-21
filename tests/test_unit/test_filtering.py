from contextlib import nullcontext as does_not_raise

import pytest
import xarray as xr

from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    log_to_attrs,
    median_filter,
    report_nan_values,
    savgol_filter,
)


def test_log_to_attrs(valid_poses_dataset):
    """Test for the ``log_to_attrs()`` decorator. Decorates a mock function and
    checks that ``attrs`` contains all expected values.
    """

    @log_to_attrs
    def fake_func(ds, arg, kwarg=None):
        return ds

    ds = fake_func(valid_poses_dataset, "test1", kwarg="test2")

    assert "log" in ds.attrs
    assert ds.attrs["log"][0]["operation"] == "fake_func"
    assert (
        ds.attrs["log"][0]["arg_1"] == "test1"
        and ds.attrs["log"][0]["kwarg"] == "test2"
    )


def test_interpolate_over_time(valid_poses_dataset_with_nan, helpers):
    """Test that the number of NaNs decreases after interpolating over time."""
    data = valid_poses_dataset_with_nan.position
    data_interp = interpolate_over_time(data)
    assert helpers.count_nans(data_interp) < helpers.count_nans(data)


def test_filter_by_confidence(valid_poses_dataset, helpers):
    """Test that points below the default 0.6 confidence threshold
    is converted to NaN.
    """
    data = valid_poses_dataset.position
    confidence = valid_poses_dataset.confidence
    data_filtered = filter_by_confidence(data, confidence)
    n_nans = helpers.count_nans(data_filtered)
    assert isinstance(data_filtered, xr.DataArray)
    # 5 timepoints * 2 individuals * 2 keypoints * 2 space dimensions
    # have confidence below 0.6
    assert n_nans == 40


@pytest.mark.parametrize("window_size", [2, 4])
def test_median_filter(valid_poses_dataset_with_nan, window_size):
    """Test that applying the median filter returns
    a different xr.DataArray than the input data.
    """
    data = valid_poses_dataset_with_nan.position
    data_smoothed = median_filter(data, window_size)
    del data_smoothed.attrs["log"]
    assert isinstance(data_smoothed, xr.DataArray) and not (
        data_smoothed.equals(data)
    )


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


def test_report_nan_values(capsys, valid_poses_dataset_with_nan):
    """Test that the correct number of NaN values are reported."""
    data = valid_poses_dataset_with_nan.position
    report_nan_values(data)
    out, _ = capsys.readouterr()
    assert data.name in out
