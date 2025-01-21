import pytest

from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    savgol_filter,
)
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
        {
            "position": filter_by_confidence(
                sample_dataset.position,
                sample_dataset.confidence,
            )
        }
    )
    n_total_nans_input = helpers.count_nans(sample_dataset.position)

    assert (
        n_total_nans_input
        == n_low_confidence_kpts * sample_dataset.sizes["space"]
    )

    # Compute maximum expected increase in NaNs due to filtering
    n_consecutive_nans_input = helpers.count_consecutive_nans(
        sample_dataset.position
    )
    max_nans_increase = (window - 1) * n_consecutive_nans_input

    # Apply savgol filter and check that number of NaNs is within threshold
    sample_dataset.update(
        {
            "position": savgol_filter(
                sample_dataset.position, window, polyorder=2
            )
        }
    )

    n_total_nans_savgol = helpers.count_nans(sample_dataset.position)

    # Check that filtering does not reduce number of nans
    assert n_total_nans_savgol >= n_total_nans_input
    # Check that the increase in nans is below the expected threshold
    assert n_total_nans_savgol - n_total_nans_input <= max_nans_increase

    # Interpolate data (without max_gap) and with extrapolation
    # and check it eliminates all NaNs
    sample_dataset.update(
        {
            "position": interpolate_over_time(
                sample_dataset.position, fill_value="extrapolate"
            )
        }
    )
    assert helpers.count_nans(sample_dataset.position) == 0
