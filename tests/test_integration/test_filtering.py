from contextlib import nullcontext as does_not_raise

import pytest
import xarray as xr

from movement.io import load_poses
from movement.sample_data import fetch_dataset_paths


@pytest.fixture
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
    sample_dataset.update(
        {"position": sample_dataset.move.filter_by_confidence()}
    )
    expected_n_nans = 13136
    n_nans_confilt = helpers.count_nans(sample_dataset.position)
    assert n_nans_confilt == expected_n_nans, (
        f"Expected {expected_n_nans} NaNs in filtered data, "
        f"got: {n_nans_confilt}"
    )
    n_consecutive_nans = helpers.count_consecutive_nans(
        sample_dataset.position
    )
    # Apply median filter and check that
    # it doesn't introduce too many or too few NaNs
    sample_dataset.update(
        {"position": sample_dataset.move.median_filter(window)}
    )
    n_nans_medfilt = helpers.count_nans(sample_dataset.position)
    max_nans_increase = (window - 1) * n_consecutive_nans
    assert (
        n_nans_medfilt <= n_nans_confilt + max_nans_increase
    ), "Median filter introduced more NaNs than expected."
    assert (
        n_nans_medfilt >= n_nans_confilt
    ), "Median filter mysteriously removed NaNs."
    n_consecutive_nans = helpers.count_consecutive_nans(
        sample_dataset.position
    )

    # Apply savgol filter and check that
    # it doesn't introduce too many or too few NaNs
    sample_dataset.update(
        {"position": sample_dataset.move.savgol_filter(window, polyorder=2)}
    )
    n_nans_savgol = helpers.count_nans(sample_dataset.position)
    max_nans_increase = (window - 1) * n_consecutive_nans
    assert (
        n_nans_savgol <= n_nans_medfilt + max_nans_increase
    ), "Savgol filter introduced more NaNs than expected."
    assert (
        n_nans_savgol >= n_nans_medfilt
    ), "Savgol filter mysteriously removed NaNs."

    # Interpolate data (without max_gap) to eliminate all NaNs
    sample_dataset.update(
        {"position": sample_dataset.move.interpolate_over_time()}
    )
    assert helpers.count_nans(sample_dataset.position) == 0


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
    return the expected data type and structure, and the
    expected ``log`` attribute containing the filtering method
    applied, if valid data variables are passed, otherwise
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
            assert result.log[0]["operation"] == method
        elif isinstance(result, dict):
            assert set(result.keys()) == set(data_vars)
            assert all(hasattr(value, "log") for value in result.values())
            assert all(
                value.log[0]["operation"] == method
                for value in result.values()
            )
