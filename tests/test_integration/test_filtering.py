import pytest

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
    return load_poses.from_dlc_file(ds_path, fps=None)


@pytest.mark.parametrize("window_length", [3, 5, 6, 13])
def test_nan_propagation_through_filters(
    sample_dataset, window_length, helpers
):
    """Tests how NaNs are propagated when passing a dataset through multiple
    filters sequentially. For the ``median_filter`` and ``savgol_filter``,
    we expect the number of NaNs to increase at most by the filter's window
    length minus one (``window_length - 1``) multiplied by the number of
    continuous stretches of NaNs present in the input dataset.
    """
    # Introduce nans via filter_by_confidence
    ds_with_nans = filter_by_confidence(sample_dataset, threshold=0.6)
    nans_after_confilt = helpers.count_nans(ds_with_nans)
    nan_repeats_after_confilt = helpers.count_nan_repeats(ds_with_nans)
    assert nans_after_confilt == 2555, (
        f"Unexpected number of NaNs in filtered dataset: "
        f"expected: 2555, got: {nans_after_confilt}"
    )

    # Apply median filter and check that
    # it doesn't introduce too many or too few NaNs
    ds_medfilt = median_filter(ds_with_nans, window_length)
    nans_after_medfilt = helpers.count_nans(ds_medfilt)
    nan_repeats_after_medfilt = helpers.count_nan_repeats(ds_medfilt)
    max_nans_increase = (window_length - 1) * nan_repeats_after_confilt
    assert (
        nans_after_medfilt <= nans_after_confilt + max_nans_increase
    ), "Median filter introduced more NaNs than expected."
    assert (
        nans_after_medfilt >= nans_after_confilt
    ), "Median filter mysteriously removed NaNs."

    # Apply savgol filter and check that
    # it doesn't introduce too many or too few NaNs
    ds_savgol = savgol_filter(
        ds_medfilt, window_length, polyorder=2, print_report=True
    )
    nans_after_savgol = helpers.count_nans(ds_savgol)
    max_nans_increase = (window_length - 1) * nan_repeats_after_medfilt
    assert (
        nans_after_savgol <= nans_after_medfilt + max_nans_increase
    ), "Savgol filter introduced more NaNs than expected."
    assert (
        nans_after_savgol >= nans_after_medfilt
    ), "Savgol filter mysteriously removed NaNs."

    # Apply interpolate_over_time (without max_gap) to eliminate all NaNs
    ds_interpolated = interpolate_over_time(ds_savgol, print_report=True)
    assert helpers.count_nans(ds_interpolated) == 0
