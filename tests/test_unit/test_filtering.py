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
list_valid_datasets_without_nans = [
    "valid_poses_dataset",
    "valid_bboxes_dataset",
]
list_valid_datasets_with_nans = [
    f"{dataset}_with_nan" for dataset in list_valid_datasets_without_nans
]
list_all_valid_datasets = (
    list_valid_datasets_without_nans + list_valid_datasets_with_nans
)


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
    list_valid_datasets_with_nans,
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
    valid_dataset = request.getfixturevalue(valid_dataset_with_nan)

    # Get position array with time unit in frames & seconds
    # assuming 10 fps = 0.1 s per frame
    position = {
        "frames": valid_dataset.position,
        "seconds": valid_dataset.position * 0.1,
    }

    # Count number of NaNs before and after interpolating position
    n_nans_before = helpers.count_nans(position["frames"])
    n_nans_after_per_time_unit = {}
    for time_unit in ["frames", "seconds"]:
        # interpolate
        position_interp = interpolate_over_time(
            position[time_unit], method="linear", max_gap=max_gap
        )
        # count nans
        n_nans_after_per_time_unit[time_unit] = helpers.count_nans(
            position_interp
        )

    # The number of NaNs should be the same for both datasets
    # as max_gap is based on number of missing observations (NaNs)
    assert (
        n_nans_after_per_time_unit["frames"]
        == n_nans_after_per_time_unit["seconds"]
    )

    # The number of NaNs should decrease after interpolation
    n_nans_after = n_nans_after_per_time_unit["frames"]
    if max_gap == 0:
        assert n_nans_after == n_nans_before
    else:
        assert n_nans_after < n_nans_before

    # The number of NaNs after interpolating should be as expected
    assert n_nans_after == (
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
    # Filter position by confidence
    valid_input_dataset = request.getfixturevalue(valid_dataset_no_nans)
    position_filtered = filter_by_confidence(
        valid_input_dataset.position,
        confidence=valid_input_dataset.confidence,
        threshold=0.6,
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
    list_all_valid_datasets,
)
@pytest.mark.parametrize(
    ("filter_func, filter_kwargs"),
    [
        (median_filter, {"window": 2}),
        (median_filter, {"window": 4}),
        (savgol_filter, {"window": 2, "polyorder": 1}),
        (savgol_filter, {"window": 4, "polyorder": 2}),
    ],
)
def test_filter_on_position(
    filter_func, filter_kwargs, valid_dataset, request
):
    """Test that applying a filter to the position data returns
    a different xr.DataArray than the input position data.
    """
    # Filter position
    valid_input_dataset = request.getfixturevalue(valid_dataset)
    position_filtered = filter_func(
        valid_input_dataset.position, **filter_kwargs
    )

    del position_filtered.attrs["log"]

    # filtered array is an xr.DataArray
    assert isinstance(position_filtered, xr.DataArray)

    # filtered data should not be equal to the original data
    assert not position_filtered.equals(valid_input_dataset.position)


@pytest.mark.parametrize(
    ("valid_dataset, expected_n_nans_in_position_per_indiv"),
    [(k, v) for k, v in expected_n_nans_in_position_per_indiv.items()],
)
@pytest.mark.parametrize(
    ("filter_func, filter_kwargs"),
    [
        (median_filter, {"window": 3}),
        (savgol_filter, {"window": 3, "polyorder": 2}),
    ],
)
def test_filter_with_nans_on_position(
    filter_func,
    filter_kwargs,
    valid_dataset,
    expected_n_nans_in_position_per_indiv,
    helpers,
    request,
):
    """Test NaN behaviour of the selected filter. The median and SG filters
    should set all values to NaN if one element of the sliding window is NaN.
    """

    def _assert_n_nans_in_position_per_individual(
        valid_input_dataset,
        position_filtered,
        expected_n_nans_in_position_per_indiv,
    ):
        # compute n nans in position after filtering per individual
        n_nans_after_filtering_per_indiv = {
            i: helpers.count_nans(position_filtered.isel(individuals=i))
            for i in range(valid_input_dataset.dims["individuals"])
        }

        # check number of nans per indiv is as expected
        for i in range(valid_input_dataset.dims["individuals"]):
            assert n_nans_after_filtering_per_indiv[i] == (
                expected_n_nans_in_position_per_indiv[i]
                * valid_input_dataset.dims["space"]
                * valid_input_dataset.dims.get("keypoints", 1)
            )

    # Filter position
    valid_input_dataset = request.getfixturevalue(valid_dataset)
    position_filtered = filter_func(
        valid_input_dataset.position, **filter_kwargs
    )

    # check number of nans per indiv is as expected
    _assert_n_nans_in_position_per_individual(
        valid_input_dataset,
        position_filtered,
        expected_n_nans_in_position_per_indiv,
    )

    # if input had nans,
    # individual 1's position at exact timepoints 0, 1 and 5 is not nan
    n_nans_input = helpers.count_nans(valid_input_dataset.position)
    if n_nans_input != 0:
        assert not (
            position_filtered.isel(individuals=0, time=[0, 1, 5])
            .isnull()
            .any()
        )


@pytest.mark.parametrize(
    "valid_dataset",
    list_all_valid_datasets,
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
