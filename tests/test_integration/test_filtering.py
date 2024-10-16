import pytest

from movement.analysis.kinematics import compute_velocity
from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    median_filter,
    savgol_filter,
)
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
    ds["velocity"] = compute_velocity(ds.position)
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
        {
            "position": filter_by_confidence(
                sample_dataset.position,
                sample_dataset.confidence,
            )
        }
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
        {"position": median_filter(sample_dataset.position, window)}
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
        {
            "position": savgol_filter(
                sample_dataset.position, window, polyorder=2
            )
        }
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
        {"position": interpolate_over_time(sample_dataset.position)}
    )
    assert helpers.count_nans(sample_dataset.position) == 0
