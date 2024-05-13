import pytest
import xarray as xr

from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    log_to_attrs,
    median_filter,
    savgol_filter,
)
from movement.sample_data import fetch_dataset


@pytest.fixture(scope="module")
def sample_dataset():
    """Return a single-animal sample dataset, with time unit in seconds."""
    return fetch_dataset("DLC_single-mouse_EPM.predictions.h5")


def test_log_to_attrs(sample_dataset):
    """Test for the ``log_to_attrs()`` decorator. Decorates a mock function and
    checks that ``attrs`` contains all expected values.
    """

    @log_to_attrs
    def fake_func(ds, arg, kwarg=None):
        return ds

    ds = fake_func(sample_dataset, "test1", kwarg="test2")

    assert "log" in ds.attrs
    assert ds.attrs["log"][0]["operation"] == "fake_func"
    assert (
        ds.attrs["log"][0]["arg_1"] == "test1"
        and ds.attrs["log"][0]["kwarg"] == "test2"
    )


def test_interpolate_over_time(sample_dataset, helpers):
    """Test the ``interpolate_over_time`` function.

    Check that the number of nans is decreased after running this function
    on a filtered dataset
    """
    ds_filtered = filter_by_confidence(sample_dataset)
    ds_interpolated = interpolate_over_time(ds_filtered)

    assert helpers.count_nans(ds_interpolated) < helpers.count_nans(
        ds_filtered
    )


def test_filter_by_confidence(sample_dataset, caplog, helpers):
    """Tests for the ``filter_by_confidence()`` function.
    Checks that the function filters the expected amount of values
    from a known dataset, and tests that this value is logged
    correctly.
    """
    ds_filtered = filter_by_confidence(sample_dataset, threshold=0.6)

    assert isinstance(ds_filtered, xr.Dataset)

    n_nans = helpers.count_nans(ds_filtered)
    assert n_nans == 2555

    # Check that diagnostics are being logged correctly
    assert f"snout: {n_nans}/{ds_filtered.time.values.shape[0]}" in caplog.text


@pytest.mark.parametrize("window_size", [0.2, 1, 4, 12])
def test_median_filter(sample_dataset, window_size):
    """Tests for the ``median_filter()`` function. Checks that
    the function successfully receives the input data and
    returns a different xr.Dataset with the correct dimensions.
    """
    ds_smoothed = median_filter(sample_dataset, window_size)

    # Test whether filter received and returned correct data
    assert isinstance(ds_smoothed, xr.Dataset) and ~(
        ds_smoothed == sample_dataset
    )
    assert ds_smoothed.position.shape == sample_dataset.position.shape


@pytest.mark.parametrize("window_length", [0.2, 1, 4, 12])
@pytest.mark.parametrize("polyorder", [1, 2, 3])
def test_savgol_filter(sample_dataset, window_length, polyorder):
    """Tests for the ``savgol_filter()`` function.
    Checks that the function successfully receives the input
    data and returns a different xr.Dataset with the correct
    dimensions.
    """
    ds_smoothed = savgol_filter(
        sample_dataset, window_length, polyorder=polyorder
    )

    # Test whether filter received and returned correct data
    assert isinstance(ds_smoothed, xr.Dataset) and ~(
        ds_smoothed == sample_dataset
    )
    assert ds_smoothed.position.shape == sample_dataset.position.shape


@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"mode": "nearest"},
        {"axis": 1},
        {"mode": "nearest", "axis": 1},
    ],
)
def test_savgol_filter_kwargs_override(sample_dataset, override_kwargs):
    """Further tests for the ``savgol_filter()`` function.
    Checks that the function raises a ValueError when the ``axis`` keyword
    argument is overridden, as this is not allowed. Overriding other keyword
    arguments (e.g. ``mode``) should not raise an error.
    """
    if "axis" in override_kwargs:
        with pytest.raises(ValueError):
            savgol_filter(sample_dataset, 5, **override_kwargs)
    else:
        ds_smoothed = savgol_filter(sample_dataset, 5, **override_kwargs)
        assert isinstance(ds_smoothed, xr.Dataset)


def test_median_filter_with_nans(valid_poses_dataset_with_nan, helpers):
    """Test nan behavior of the ``median_filter()`` function. The
    ``valid_poses_dataset_with_nan`` dataset (fixture defined in conftest.py)
    contains NaN values in all keypoints of the first individual at times
    3, 7, and 8 (0-indexed, 10 total timepoints).
    The median filter should propagate NaNs within the windows of the filter,
    but it should not introduce any NaNs for the second individual.
    """
    ds_smoothed = median_filter(valid_poses_dataset_with_nan, 3)
    # There should be NaNs at 7 timepoints for the first individual
    # all except for timepoints 0, 1 and 5
    assert helpers.count_nans(ds_smoothed) == 7
    assert (
        ~ds_smoothed.position.isel(individuals=0, time=[0, 1, 5])
        .isnull()
        .any()
    )
    # The second individual should not contain any NaNs
    assert ~ds_smoothed.position.sel(individuals="ind2").isnull().any()
