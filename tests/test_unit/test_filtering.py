from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.filtering import (
    filter_by_confidence,
    filter_short_trajectories,
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


# -------------------- Tests for filter_short_trajectories --------------------
class TestFilterShortTrajectories:
    """Test filter_short_trajectories on valid datasets."""

    @pytest.fixture
    def poses_dataset_varied_lengths(self):
        """Poses dataset with individuals having different trajectory lengths.

        - id_0: 90 valid frames out of 100
        - id_1: 50 valid frames out of 100
        - id_2: 10 valid frames out of 100
        """
        n_time, n_space, n_keypoints, n_individuals = 100, 2, 3, 3
        rng = np.random.default_rng(42)
        position = rng.random((n_time, n_space, n_keypoints, n_individuals))
        position[90:, :, :, 0] = np.nan
        position[50:, :, :, 1] = np.nan
        position[10:, :, :, 2] = np.nan
        confidence = rng.random((n_time, n_keypoints, n_individuals))

        return xr.Dataset(
            data_vars={
                "position": xr.DataArray(
                    position,
                    dims=("time", "space", "keypoints", "individuals"),
                ),
                "confidence": xr.DataArray(
                    confidence,
                    dims=("time", "keypoints", "individuals"),
                ),
            },
            coords={
                "time": np.arange(n_time),
                "space": ["x", "y"],
                "keypoints": ["snout", "center", "tail"],
                "individuals": ["id_0", "id_1", "id_2"],
            },
            attrs={"ds_type": "poses", "source_software": "test"},
        )

    @pytest.fixture
    def bboxes_dataset_varied_lengths(self):
        """Bboxes dataset with individuals having different trajectory lengths.

        - crab_0: 80 valid frames out of 100
        - crab_1: 100 valid frames out of 100
        - crab_2: 20 valid frames out of 100
        """
        n_time, n_space, n_individuals = 100, 2, 3
        rng = np.random.default_rng(42)
        position = rng.random((n_time, n_space, n_individuals))
        position[80:, :, 0] = np.nan
        position[20:, :, 2] = np.nan
        confidence = rng.random((n_time, n_individuals))

        return xr.Dataset(
            data_vars={
                "position": xr.DataArray(
                    position,
                    dims=("time", "space", "individuals"),
                ),
                "confidence": xr.DataArray(
                    confidence,
                    dims=("time", "individuals"),
                ),
            },
            coords={
                "time": np.arange(n_time),
                "space": ["x", "y"],
                "individuals": ["crab_0", "crab_1", "crab_2"],
            },
            attrs={"ds_type": "bboxes", "source_software": "test"},
        )

    def test_filter_poses_basic(self, poses_dataset_varied_lengths):
        """Test basic filtering keeps only individuals above threshold."""
        result = filter_short_trajectories(
            poses_dataset_varied_lengths, min_valid_frames=80
        )
        assert len(result.individuals) == 1
        assert "id_0" in result.individuals.values

    def test_filter_bboxes_basic(self, bboxes_dataset_varied_lengths):
        """Test filtering on bboxes dataset."""
        result = filter_short_trajectories(
            bboxes_dataset_varied_lengths, min_valid_frames=50
        )
        assert len(result.individuals) == 2
        assert "crab_0" in result.individuals.values
        assert "crab_1" in result.individuals.values
        assert "crab_2" not in result.individuals.values

    def test_filter_none_removed(self, poses_dataset_varied_lengths):
        """Test when no individuals are filtered."""
        result = filter_short_trajectories(
            poses_dataset_varied_lengths, min_valid_frames=5
        )
        assert len(result.individuals) == 3

    def test_filter_all_removed(self, poses_dataset_varied_lengths):
        """Test when all individuals are filtered."""
        result = filter_short_trajectories(
            poses_dataset_varied_lengths, min_valid_frames=200
        )
        assert len(result.individuals) == 0

    def test_edge_case_exact_threshold(self, poses_dataset_varied_lengths):
        """Individual with exactly min_valid_frames should be kept."""
        result = filter_short_trajectories(
            poses_dataset_varied_lengths, min_valid_frames=90
        )
        assert "id_0" in result.individuals.values

    def test_confidence_also_filtered(self, poses_dataset_varied_lengths):
        """All data variables should be filtered, not just position."""
        result = filter_short_trajectories(
            poses_dataset_varied_lengths, min_valid_frames=80
        )
        assert "confidence" in result.data_vars
        assert result.confidence.sizes["individuals"] == 1

    def test_preserves_attributes(self, poses_dataset_varied_lengths):
        """Dataset attributes should be preserved."""
        result = filter_short_trajectories(
            poses_dataset_varied_lengths, min_valid_frames=50
        )
        assert result.attrs["ds_type"] == "poses"

    def test_preserves_coordinates(self, poses_dataset_varied_lengths):
        """Non-individuals coordinates should be unchanged."""
        result = filter_short_trajectories(
            poses_dataset_varied_lengths, min_valid_frames=50
        )
        np.testing.assert_array_equal(
            result.time.values, poses_dataset_varied_lengths.time.values
        )
        np.testing.assert_array_equal(
            result.space.values, poses_dataset_varied_lengths.space.values
        )

    @pytest.mark.parametrize(
        "bad_value",
        [-1, 0, 10.5, "abc"],
    )
    def test_invalid_min_valid_frames(
        self, poses_dataset_varied_lengths, bad_value
    ):
        """Non-positive or non-integer values should raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            filter_short_trajectories(
                poses_dataset_varied_lengths, min_valid_frames=bad_value
            )

    def test_no_individuals_dimension(self):
        """Dataset without individuals dimension should raise ValueError."""
        ds = xr.Dataset(
            {
                "position": xr.DataArray(
                    np.random.rand(10, 2), dims=("time", "space")
                )
            },
        )
        with pytest.raises(ValueError, match="individuals"):
            filter_short_trajectories(ds, min_valid_frames=5)

    def test_print_report(self, poses_dataset_varied_lengths, capsys):
        """print_report should output filtering summary."""
        filter_short_trajectories(
            poses_dataset_varied_lengths,
            min_valid_frames=50,
            print_report=True,
        )
        captured = capsys.readouterr()
        assert "Filtered" in captured.out
        assert "valid frames" in captured.out

    def test_single_individual_kept(self, valid_poses_dataset):
        """Test on standard fixture (10 frames, 2 individuals, no NaNs)."""
        result = filter_short_trajectories(
            valid_poses_dataset, min_valid_frames=10
        )
        assert len(result.individuals) == 2

    def test_single_individual_dataset(self):
        """Dataset with only 1 individual should work correctly."""
        ds = xr.Dataset(
            {
                "position": xr.DataArray(
                    np.random.rand(10, 2, 1),
                    dims=("time", "space", "individuals"),
                    coords={"individuals": ["solo"]},
                )
            },
        )
        result = filter_short_trajectories(ds, min_valid_frames=5)
        assert len(result.individuals) == 1

    def test_all_nan_individual(self):
        """Individual with all NaN frames should be filtered."""
        position = np.random.rand(10, 2, 2)
        position[:, :, 1] = np.nan
        ds = xr.Dataset(
            {
                "position": xr.DataArray(
                    position,
                    dims=("time", "space", "individuals"),
                    coords={"individuals": ["valid", "empty"]},
                )
            },
        )
        result = filter_short_trajectories(ds, min_valid_frames=1)
        assert len(result.individuals) == 1
        assert "valid" in result.individuals.values
