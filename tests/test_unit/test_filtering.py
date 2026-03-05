from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.filtering import (
    filter_by_confidence,
    filter_valid_regions,
    interpolate_over_time,
    rolling_filter,
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


@pytest.mark.parametrize(
    "valid_dataset",
    list_all_valid_datasets,
)
class TestFilteringValidDataset:
    """Test rolling and savgol filtering on
    valid datasets with/without NaNs.
    """

    @pytest.mark.parametrize(
        ("filter_func, filter_kwargs"),
        [
            (rolling_filter, {"window": 3, "statistic": "median"}),
            (savgol_filter, {"window": 3, "polyorder": 2, "mode": "nearest"}),
        ],
    )
    def test_filter_with_nans_on_position(
        self, filter_func, filter_kwargs, valid_dataset, helpers, request
    ):
        """Test NaN behaviour of the rolling and SG filters.
        Both filters should set all values to NaN if one element of the
        sliding window is NaN.
        """
        # Expected number of nans in the position array per individual
        expected_nans_in_filtered_position_per_indiv = {
            "valid_poses_dataset": [0, 0],  # no nans in input
            "valid_bboxes_dataset": [0, 0],  # no nans in input
            "valid_poses_dataset_with_nan": [38, 0],
            "valid_bboxes_dataset_with_nan": [14, 0],
        }
        # Filter position
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        position_filtered = filter_func(
            valid_input_dataset.position, **filter_kwargs, print_report=True
        )
        # Compute n nans in position after filtering per individual
        n_nans_after_filtering_per_indiv = [
            helpers.count_nans(position_filtered.isel(individuals=i))
            for i in range(valid_input_dataset.sizes["individuals"])
        ]
        # Check number of nans per indiv is as expected
        assert (
            n_nans_after_filtering_per_indiv
            == expected_nans_in_filtered_position_per_indiv[valid_dataset]
        )

    @pytest.mark.parametrize(
        "override_kwargs, expected_exception",
        [
            ({"mode": "nearest", "print_report": True}, does_not_raise()),
            ({"axis": 1}, pytest.raises(ValueError)),
            ({"mode": "nearest", "axis": 1}, pytest.raises(ValueError)),
            (  # polyorder >= window: re-raised unchanged by savgol_filter
                {"polyorder": 5},
                pytest.raises(ValueError, match="polyorder"),
            ),
        ],
    )
    def test_savgol_filter_kwargs_override(
        self, valid_dataset, override_kwargs, expected_exception, request
    ):
        """Test that overriding keyword arguments in the
        Savitzky-Golay filter works, except for the ``axis`` argument,
        which should raise a ValueError.
        """
        with expected_exception:
            savgol_filter(
                request.getfixturevalue(valid_dataset).position,
                window=3,
                **override_kwargs,
            )

    @pytest.mark.parametrize(
        "statistic, expected_exception",
        [
            ("mean", does_not_raise()),
            ("median", does_not_raise()),
            ("max", does_not_raise()),
            ("min", does_not_raise()),
            ("invalid", pytest.raises(ValueError, match="Invalid statistic")),
        ],
    )
    def test_rolling_filter_statistic(
        self, valid_dataset, statistic, expected_exception, request
    ):
        """Test that the rolling filter works with different statistics."""
        with expected_exception:
            rolling_filter(
                request.getfixturevalue(valid_dataset).position,
                window=3,
                statistic=statistic,
            )


@pytest.mark.parametrize(
    "valid_dataset_with_nan",
    list_valid_datasets_with_nans,
)
class TestFilteringValidDatasetWithNaNs:
    """Test filtering functions on datasets with NaNs."""

    @pytest.mark.parametrize(
        "max_gap, expected_n_nans_in_position",
        [(None, [22, 0]), (0, [28, 6]), (1, [26, 4]), (2, [22, 0])],
        # expected total n nans: [poses, bboxes]
    )
    def test_interpolate_over_time_on_position(
        self,
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
        valid_dataset_in_frames = request.getfixturevalue(
            valid_dataset_with_nan
        )
        # Get position array with time unit in frames & seconds
        # assuming 10 fps = 0.1 s per frame
        valid_dataset_in_seconds = valid_dataset_in_frames.copy()
        valid_dataset_in_seconds.coords["time"] = (
            valid_dataset_in_seconds.coords["time"] * 0.1
        )
        position = {
            "frames": valid_dataset_in_frames.position,
            "seconds": valid_dataset_in_seconds.position,
        }
        # Count number of NaNs
        n_nans_after_per_time_unit = {}
        for time_unit in ["frames", "seconds"]:
            # interpolate
            position_interp = interpolate_over_time(
                position[time_unit],
                method="linear",
                max_gap=max_gap,
                print_report=True,
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
        # The number of NaNs after interpolating should be as expected
        n_nans_after = n_nans_after_per_time_unit["frames"]
        dataset_index = list_valid_datasets_with_nans.index(
            valid_dataset_with_nan
        )
        assert n_nans_after == expected_n_nans_in_position[dataset_index]

    @pytest.mark.parametrize(
        "window",
        [3, 5, 6, 10],  # input data has 10 frames
    )
    @pytest.mark.parametrize("filter_func", [rolling_filter, savgol_filter])
    def test_filter_with_nans_on_position_varying_window(
        self, valid_dataset_with_nan, window, filter_func, helpers, request
    ):
        """Test that the number of NaNs in the filtered position data
        increases at most by the filter's window length minus one
        multiplied by the number of consecutive NaNs in the input data.
        """
        # Prepare kwargs per filter
        kwargs = {"window": window}
        if filter_func == savgol_filter:
            kwargs["polyorder"] = 2
            # Use 'nearest' to avoid edge NaNs errors with 'interp' mode
            kwargs["mode"] = "nearest"

        valid_input_dataset = request.getfixturevalue(valid_dataset_with_nan)
        position_filtered = filter_func(
            valid_input_dataset.position,
            **kwargs,
        )
        # Count number of NaNs in the input and filtered position data
        n_total_nans_initial = helpers.count_nans(valid_input_dataset.position)
        n_consecutive_nans_initial = helpers.count_consecutive_nans(
            valid_input_dataset.position
        )
        n_total_nans_filtered = helpers.count_nans(position_filtered)
        max_nans_increase = (window - 1) * n_consecutive_nans_initial
        # Check that filtering does not reduce number of nans
        assert n_total_nans_filtered >= n_total_nans_initial
        # Check that the increase in nans is below the expected threshold
        assert (
            n_total_nans_filtered - n_total_nans_initial <= max_nans_increase
        )

    @pytest.mark.parametrize(
        "mode, expected_exception",
        [
            (None, pytest.raises(ValueError, match="mode='interp'")),
            ("interp", pytest.raises(ValueError, match="mode='interp'")),
            ("nearest", does_not_raise()),
            ("mirror", does_not_raise()),
        ],
        ids=[
            "default mode (interp): raise error",
            "explicitly pass mode=interp: raise error",
            "explicitly pass mode=nearest: no error",
            "explicitly pass mode=mirror: no error",
        ],
    )
    def test_savgol_filter_mode_edge_nans(
        self, valid_dataset_with_nan, mode, expected_exception, request
    ):
        """Test savgol_filter edge-NaN behavior across modes.

        When mode is 'interp' and there are NaNs in the edge windows
        (which is the case for both types of valid_dataset_with_nan),
        a ValueError should be raised.
        """
        dataset = request.getfixturevalue(valid_dataset_with_nan)
        with expected_exception:
            kwargs = {"window": 3, "polyorder": 2}
            if mode is not None:
                kwargs["mode"] = mode
            savgol_filter(dataset.position, **kwargs)


@pytest.mark.parametrize(
    "valid_dataset_no_nans",
    list_valid_datasets_without_nans,
)
def test_filter_by_confidence_on_position(
    valid_dataset_no_nans, helpers, request
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
        print_report=True,
    )
    # Count number of NaNs in the full array
    n_nans = helpers.count_nans(position_filtered)
    # expected number of nans for poses:
    # 5 timepoints * 2 individuals * 2 keypoints
    # Note: we count the number of nans in the array, so we multiply
    # the number of low confidence keypoints by the number of
    # space dimensions
    n_low_confidence_kpts = 5
    assert isinstance(position_filtered, xr.DataArray)
    assert n_nans == valid_input_dataset.sizes["space"] * n_low_confidence_kpts


# ---------------------------------------------------------------------------
# Helpers for TestFilterValidRegions
# ---------------------------------------------------------------------------


def _make_da(n_frames: int, nan_at: list[int] | None = None) -> xr.DataArray:
    """Return a (time, space) DataArray with optional all-NaN time steps."""
    data = np.ones((n_frames, 2), dtype=float)
    if nan_at:
        for t in nan_at:
            data[t] = np.nan
    return xr.DataArray(
        data,
        dims=["time", "space"],
        coords={"time": np.arange(n_frames), "space": ["x", "y"]},
    )


class TestFilterValidRegions:
    """Tests for :func:`movement.filtering.filter_valid_regions`."""

    def test_clean_data_returns_full_array(self):
        """All frames valid → entire array is returned."""
        da = _make_da(10)
        result = filter_valid_regions(da)
        assert result.sizes["time"] == 10
        assert list(result.time.values) == list(range(10))

    def test_returns_longest_valid_run(self):
        """NaNs at frames 3-5 split data; longest run (6-9) returned."""
        da = _make_da(10, nan_at=[3, 4, 5])
        result = filter_valid_regions(da)
        assert result.sizes["time"] == 4
        assert list(result.time.values) == [6, 7, 8, 9]

    def test_max_nan_fraction_allows_partial_nan_frame(self):
        """A frame with 50% NaN is valid when max_nan_fraction=0.5."""
        da = _make_da(5)
        da.values[2, 0] = np.nan  # x at time=2 → 1/2 = 50% NaN
        result = filter_valid_regions(da, max_nan_fraction=0.5)
        assert result.sizes["time"] == 5

    def test_time_coordinates_preserved(self):
        """Returned slice carries the correct original time coordinates."""
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        data = np.ones((5, 2), dtype=float)
        data[1] = np.nan  # frame 1 NaN
        data[2] = np.nan  # frame 2 NaN
        da = xr.DataArray(
            data,
            dims=["time", "space"],
            coords={"time": times, "space": ["x", "y"]},
        )
        # valid runs: frame 0 (len 1), frames 3-4 (len 2)
        result = filter_valid_regions(da, min_length=2)
        assert result.sizes["time"] == 2
        np.testing.assert_allclose(result.time.values, [0.3, 0.4])

    def test_no_valid_region_raises(self):
        """All frames NaN → ValueError because longest run < min_length."""
        da = _make_da(4, nan_at=[0, 1, 2, 3])
        with pytest.raises(ValueError, match="No valid region"):
            filter_valid_regions(da)

    def test_min_length_exceeds_best_run_raises(self):
        """Longest valid run (4 frames) < min_length=5 → ValueError."""
        da = _make_da(10, nan_at=[3, 4, 5])
        with pytest.raises(ValueError, match="No valid region"):
            filter_valid_regions(da, min_length=5)

    @pytest.mark.parametrize(
        "bad_fraction, match_str",
        [
            (-0.1, "max_nan_fraction"),
            (1.5, "max_nan_fraction"),
        ],
    )
    def test_invalid_max_nan_fraction_raises(self, bad_fraction, match_str):
        """max_nan_fraction outside [0, 1] raises ValueError."""
        da = _make_da(5)
        with pytest.raises(ValueError, match=match_str):
            filter_valid_regions(da, max_nan_fraction=bad_fraction)

    def test_invalid_min_length_raises(self):
        """min_length < 1 raises ValueError."""
        da = _make_da(5)
        with pytest.raises(ValueError, match="min_length"):
            filter_valid_regions(da, min_length=0)

    def test_result_is_dataarray(self):
        """Return type is always xarray.DataArray."""
        da = _make_da(5)
        result = filter_valid_regions(da)
        assert isinstance(result, xr.DataArray)
