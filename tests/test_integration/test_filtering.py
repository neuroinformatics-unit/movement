from contextlib import nullcontext as does_not_raise

import pytest
import xarray as xr

from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    median_filter,
    savgol_filter,
)
from movement.io import load_poses
from movement.sample_data import fetch_dataset_paths


@pytest.fixture(scope="module")
def sample_dataset():
    """Return a single-animal sample dataset, with time unit in frames.
    This allows us to better control the expected number of NaNs in the tests.
    """
    ds_path = fetch_dataset_paths("DLC_single-mouse_EPM.predictions.h5")[
        "poses"
    ]
    ds = load_poses.from_dlc_file(ds_path)
    ds["velocity"] = ds.move.compute_velocity()
    return ds


@pytest.mark.parametrize("window", [3, 5, 6, 13])
def test_nan_propagation_through_filters(sample_dataset, window, helpers):
    """Test NaN propagation when passing a DataArray through
    multiple filters sequentially. For the ``median_filter``
    and ``savgol_filter``, the number of NaNs is expected to increase
    at most by the filter's window length minus one (``window - 1``)
    multiplied by the number of consecutive NaNs in the input data.
    """
    # Introduce nans via filter_by_confidence
    data = sample_dataset.position
    confidence = sample_dataset.confidence
    data_confilt = filter_by_confidence(data, confidence)
    expected_n_nans = 13136
    n_nans_confilt = helpers.count_nans(data_confilt)
    assert n_nans_confilt == expected_n_nans, (
        f"Expected {expected_n_nans} NaNs in filtered data, "
        f"got: {n_nans_confilt}"
    )
    n_consecutive_nans = helpers.count_consecutive_nans(data_confilt)
    # Apply median filter and check that
    # it doesn't introduce too many or too few NaNs
    data_medfilt = median_filter(data_confilt, window)
    n_nans_medfilt = helpers.count_nans(data_medfilt)
    max_nans_increase = (window - 1) * n_consecutive_nans
    assert (
        n_nans_medfilt <= n_nans_confilt + max_nans_increase
    ), "Median filter introduced more NaNs than expected."
    assert (
        n_nans_medfilt >= n_nans_confilt
    ), "Median filter mysteriously removed NaNs."
    n_consecutive_nans = helpers.count_consecutive_nans(data_medfilt)

    # Apply savgol filter and check that
    # it doesn't introduce too many or too few NaNs
    data_savgol = savgol_filter(data_medfilt, window, polyorder=2)
    n_nans_savgol = helpers.count_nans(data_savgol)
    max_nans_increase = (window - 1) * n_consecutive_nans
    assert (
        n_nans_savgol <= n_nans_medfilt + max_nans_increase
    ), "Savgol filter introduced more NaNs than expected."
    assert (
        n_nans_savgol >= n_nans_medfilt
    ), "Savgol filter mysteriously removed NaNs."

    # Interpolate data (without max_gap) to eliminate all NaNs
    data_interp = interpolate_over_time(data_savgol)
    assert helpers.count_nans(data_interp) == 0


@pytest.mark.parametrize(
    "method",
    [
        "filter_by_confidence",
        "interpolate_over_time",
        "median_filter",
        "savgol_filter",
    ],
)
@pytest.mark.parametrize(
    "data_vars, expected_exception",
    [
        (None, does_not_raise(xr.DataArray)),
        (["position", "velocity"], does_not_raise(dict)),
        (["vlocity"], pytest.raises(RuntimeError)),  # Does not exist
    ],
)
def test_accessor_filter_method(
    sample_dataset, method, data_vars, expected_exception
):
    """Test that filtering methods in the ``move`` accessor
    return the expected data type and structure, with the
    expected ``log`` attribute added by the filtering methods
    if valid data variables are passed, otherwise
    raise an exception.
    """
    with expected_exception as expected_type:
        if method in ["median_filter", "savgol_filter"]:
            # supply required "window" argument
            result = getattr(sample_dataset.move, method)(
                data_vars=data_vars, window=3
            )
        else:
            result = getattr(sample_dataset.move, method)(data_vars=data_vars)
        assert isinstance(result, expected_type)
        if isinstance(result, xr.DataArray):
            assert hasattr(result, "log")
        elif isinstance(result, dict):
            assert set(result.keys()) == set(data_vars)
            assert all(hasattr(value, "log") for value in result.values())
