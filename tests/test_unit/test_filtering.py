from contextlib import nullcontext as does_not_raise

import pytest
import xarray as xr

from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    median_filter,
    savgol_filter,
)

# Dataset fixtures
valid_datasets_without_nans = [
    "valid_poses_dataset",
    "valid_bboxes_dataset",
]
valid_datasets_with_nans = [
    f"{dataset}_with_nan" for dataset in valid_datasets_without_nans
]
all_valid_datasets = valid_datasets_without_nans + valid_datasets_with_nans


# Expected number of nans in the position array per individual,
# for each dataset
expected_n_nans_in_position_per_indiv = {
    "valid_poses_dataset": {0: 0, 1: 0},
    # filtering should not introduce nans if input has no nans
    "valid_bboxes_dataset": {0: 0, 1: 0},
    # filtering should not introduce nans if input has no nans
    "valid_poses_dataset_with_nan": {0: 7, 1: 0},
    # individual with index 0 has 7 frames with nans in position after
    # filtering individual with index 1 has no nans after filtering
    "valid_bboxes_dataset_with_nan": {0: 7, 1: 0},
    # individual with index 0 has 7 frames with nans in position after
    # filtering individual with index 0 has no nans after filtering
}


@pytest.mark.parametrize(
    "valid_dataset_with_nan",
    valid_datasets_with_nans,
)
@pytest.mark.parametrize(
    "max_gap, expected_n_nans_in_position", [(None, 0), (0, 3), (1, 2), (2, 0)]
)
def test_interpolate_over_time_on_position(
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
    assert n_nans_after_frames == (
        valid_dataset.dims["space"]
        * valid_dataset.dims.get("keypoints", 1)
        # in bboxes dataset there is no keypoints dimension
        * expected_n_nans_in_position
    )


@pytest.mark.parametrize(
    "valid_dataset_no_nans, n_low_confidence_kpts",
    [
        ("valid_poses_dataset", 20),
        ("valid_bboxes_dataset", 5),
    ],
)
def test_filter_by_confidence_on_position(
    valid_dataset_no_nans, n_low_confidence_kpts, helpers, request
):
    """Test that points below the default 0.6 confidence threshold
    are converted to NaN.
    """
    valid_input_dataset = request.getfixturevalue(valid_dataset_no_nans)

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
    "valid_dataset",
    all_valid_datasets,
)
@pytest.mark.parametrize("window_size", [2, 4])
def test_median_filter_on_position(valid_dataset, window_size, request):
    """Test that applying the median filter to the position data returns
    a different xr.DataArray than the input position data.
    """
    valid_input_dataset = request.getfixturevalue(valid_dataset)

    position = valid_input_dataset.position
    position_filtered = median_filter(position, window_size)

    del position_filtered.attrs["log"]

    # filtered array is an xr.DataArray
    assert isinstance(position_filtered, xr.DataArray)

    # filtered data should not be equal to the original data
    assert not position_filtered.equals(position)


@pytest.mark.parametrize(
    ("valid_dataset, expected_n_nans_in_position_per_indiv"),
    [(k, v) for k, v in expected_n_nans_in_position_per_indiv.items()],
)
def test_median_filter_with_nans_on_position(
    valid_dataset,
    expected_n_nans_in_position_per_indiv,
    helpers,
    request,
):
    """Test NaN behaviour of the median filter. The median filter
    should propagate NaNs within the windows of the filter.
    """
    # get input data
    valid_input_dataset = request.getfixturevalue(valid_dataset)
    position = valid_input_dataset.position

    # apply median filter
    position_filtered = median_filter(position, window=3)

    # count nans in input
    n_nans_input = helpers.count_nans(position)

    # count nans after filtering per individual
    n_nans_after_filtering_per_indiv = {
        i: helpers.count_nans(position_filtered.isel(individuals=i))
        for i in range(valid_input_dataset.dims["individuals"])
    }

    # check number of nans is as expected
    for i in range(valid_input_dataset.dims["individuals"]):
        assert n_nans_after_filtering_per_indiv[i] == (
            expected_n_nans_in_position_per_indiv[i]
            * valid_input_dataset.dims["space"]
            * valid_input_dataset.dims.get("keypoints", 1)
        )

    if n_nans_input != 0:
        # individual 1's position at exact timepoints 0, 1 and 5 is not nan
        assert not (
            position_filtered.isel(individuals=0, time=[0, 1, 5])
            .isnull()
            .any()
        )


@pytest.mark.parametrize(
    "valid_dataset",
    all_valid_datasets,
)
@pytest.mark.parametrize("window, polyorder", [(2, 1), (4, 2)])
def test_savgol_filter_on_position(valid_dataset, window, polyorder, request):
    """Test that applying the Savitzky-Golay filter to the position data
    returns a different xr.DataArray than the input position data.
    """
    valid_input_dataset = request.getfixturevalue(valid_dataset)

    position = valid_input_dataset.position
    posiiton_filtered = savgol_filter(
        position, window=window, polyorder=polyorder
    )

    del posiiton_filtered.attrs["log"]

    # filtered array is an xr.DataArray
    assert isinstance(posiiton_filtered, xr.DataArray)

    # filtered data should not be equal to the original data
    assert not (posiiton_filtered.equals(position))


@pytest.mark.parametrize(
    ("valid_dataset, expected_n_nans_in_position_per_indiv"),
    [(k, v) for k, v in expected_n_nans_in_position_per_indiv.items()],
)
def test_savgol_filter_with_nans_on_position(
    valid_dataset, expected_n_nans_in_position_per_indiv, helpers, request
):
    """Test NaN behaviour of the Savitzky-Golay filter. The Savitzky-Golay
    filter should propagate NaNs within the windows of the filter.
    """
    # get input data
    valid_input_dataset = request.getfixturevalue(valid_dataset)
    position = valid_input_dataset.position

    # apply SG filter
    position_filtered = savgol_filter(position, window=3, polyorder=2)

    # count nans in input
    n_nans_input = helpers.count_nans(position)

    # count nans after filtering per individual
    n_nans_after_filtering_per_indiv = {
        i: helpers.count_nans(position_filtered.isel(individuals=i))
        for i in range(valid_input_dataset.dims["individuals"])
    }

    # check number of nans is as expected
    for i in range(valid_input_dataset.dims["individuals"]):
        assert n_nans_after_filtering_per_indiv[i] == (
            expected_n_nans_in_position_per_indiv[i]
            * valid_input_dataset.dims["space"]
            * valid_input_dataset.dims.get("keypoints", 1)
        )

    # if input data had nans, the filtered position of individual 1
    # will be nan at every timepoint except for timepoints 0, 1 and 5
    if n_nans_input != 0:
        # individual 1's position at exact timepoints 0, 1 and 5 is not nan
        assert not (
            position_filtered.isel(individuals=0, time=[0, 1, 5])
            .isnull()
            .any()
        )


@pytest.mark.parametrize(
    "valid_dataset",
    all_valid_datasets,
)
@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"mode": "nearest"},
        {"axis": 1},
        {"mode": "nearest", "axis": 1},
    ],
)
def test_savgol_filter_kwargs_override(
    valid_dataset, override_kwargs, request
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
            request.getfixturevalue(valid_dataset).position,
            window=3,
            **override_kwargs,
        )
