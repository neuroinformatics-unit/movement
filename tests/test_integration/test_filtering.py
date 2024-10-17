from contextlib import nullcontext as does_not_raise

import pytest
import xarray as xr

from movement.io import load_poses
from movement.sample_data import fetch_dataset_paths


@pytest.fixture
def sample_dataset():
    """Return a single-animal sample dataset, with time unit in frames."""
    ds_path = fetch_dataset_paths("DLC_single-mouse_EPM.predictions.h5")[
        "poses"
    ]
    ds = load_poses.from_dlc_file(ds_path)

    return ds


@pytest.mark.parametrize("window", [3, 5, 6, 13])
def test_nan_propagation_through_filters(sample_dataset, window, helpers):
    """Test NaN propagation is as expected when passing a DataArray through
    filter by confidence, Savgol filter and interpolation.
    For the ``savgol_filter``, the number of NaNs is expected to increase
    at most by the filter's window length minus one (``window - 1``)
    multiplied by the number of consecutive NaNs in the input data.
    """
    # Compute number of low confidence keypoints
    n_low_confidence_kpts = (sample_dataset.confidence.data < 0.6).sum()

    # Check filter position by confidence creates correct number of NaNs
    sample_dataset.update(
        {"position": sample_dataset.move.filter_by_confidence()}
    )
    n_total_nans_input = helpers.count_nans(sample_dataset.position)

    assert (
        n_total_nans_input
        == n_low_confidence_kpts * sample_dataset.dims["space"]
    )

    # Compute maximum expected increase in NaNs due to filtering
    n_consecutive_nans_input = helpers.count_consecutive_nans(
        sample_dataset.position
    )
    max_nans_increase = (window - 1) * n_consecutive_nans_input

    # Apply savgol filter and check that number of NaNs is within threshold
    sample_dataset.update(
        {"position": sample_dataset.move.savgol_filter(window, polyorder=2)}
    )

    n_total_nans_savgol = helpers.count_nans(sample_dataset.position)

    # Check that filtering does not reduce number of nans
    assert n_total_nans_savgol >= n_total_nans_input
    # Check that the increase in nans is below the expected threshold
    assert n_total_nans_savgol - n_total_nans_input <= max_nans_increase

    # Interpolate data (without max_gap) and check it eliminates all NaNs
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
    # Compute velocity
    sample_dataset["velocity"] = sample_dataset.move.compute_velocity()

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
