"""Tests for split_trajectories_by_gap function."""

import numpy as np
import pytest
import xarray as xr

from movement.trajectory_splitting import split_trajectories_by_gap


def _create_test_position_data(
    n_frames: int = 6,
    n_space: int = 2,
    n_keypoints: int = 2,
    n_individuals: int = 1,
    individual_names: list[str] | None = None,
) -> xr.DataArray:
    """Create a synthetic position DataArray for testing.

    Parameters
    ----------
    n_frames : int
        Number of time frames.
    n_space : int
        Number of spatial dimensions (2 for x,y or 3 for x,y,z).
    n_keypoints : int
        Number of keypoints.
    n_individuals : int
        Number of individuals.
    individual_names : list of str, optional
        Custom names for individuals. Defaults to ["mouse0", "mouse1", ...].

    Returns
    -------
    xr.DataArray
        A DataArray with dimensions (time, space, keypoints, individuals)
        filled with sequential values starting from 1.

    """
    if individual_names is None:
        individual_names = [f"mouse{i}" for i in range(n_individuals)]

    # Create data filled with ones (non-NaN) for simplicity
    data = np.ones(
        (n_frames, n_space, n_keypoints, n_individuals), dtype=float
    )

    space_labels = ["x", "y", "z"][:n_space]
    keypoint_labels = [f"kp{i}" for i in range(n_keypoints)]

    return xr.DataArray(
        data,
        dims=("time", "space", "keypoints", "individuals"),
        coords={
            "time": np.arange(n_frames),
            "space": space_labels,
            "keypoints": keypoint_labels,
            "individuals": individual_names,
        },
        attrs={"source_software": "test", "fps": 30},
    )


class TestSplitTrajectoriesByGap:
    """Test suite for split_trajectories_by_gap function."""

    def test_basic_split_with_gap(self):
        """Test basic split: 1 individual, 2 keypoints, 2 spaces, 6 frames.

        Insert NaNs at frames 2-3, min_gap_size=1.
        Expected: Two segments - frames [0-1] and [4-5].

        Time:  0 1 2 3 4 5
        Valid: T T F F T T
        """
        position = _create_test_position_data(
            n_frames=6,
            n_space=2,
            n_keypoints=2,
            n_individuals=1,
            individual_names=["mouse0"],
        )

        # Insert NaNs at frames 2 and 3
        position.loc[{"time": [2, 3]}] = np.nan

        result = split_trajectories_by_gap(position, min_gap_size=1)

        # Should have 2 individuals now
        assert result.sizes["individuals"] == 2
        assert list(result.coords["individuals"].values) == [
            "mouse0_000",
            "mouse0_001",
        ]

        # First segment: valid at frames 0-1, NaN elsewhere
        seg0 = result.sel(individuals="mouse0_000")
        assert not np.isnan(seg0.sel(time=[0, 1]).values).any()
        assert np.isnan(seg0.sel(time=[2, 3, 4, 5]).values).all()

        # Second segment: valid at frames 4-5, NaN elsewhere
        seg1 = result.sel(individuals="mouse0_001")
        assert not np.isnan(seg1.sel(time=[4, 5]).values).any()
        assert np.isnan(seg1.sel(time=[0, 1, 2, 3]).values).all()

    def test_min_gap_size_threshold(self):
        """Test that min_gap_size correctly controls splitting.

        Time:  0 1 2 3 4 5
        Valid: T T F F T T  (gap of length 2)

        - min_gap_size=2: Should split (gap=2 >= threshold=2)
        - min_gap_size=3: Should NOT split (gap=2 < threshold=3)
        """
        position = _create_test_position_data(
            n_frames=6,
            n_space=2,
            n_keypoints=2,
            n_individuals=1,
            individual_names=["mouse0"],
        )

        # Insert NaNs at frames 2 and 3 (gap of length 2)
        position.loc[{"time": [2, 3]}] = np.nan

        # With min_gap_size=2, gap=2 is enough to split
        result_split = split_trajectories_by_gap(position, min_gap_size=2)
        assert result_split.sizes["individuals"] == 2

        # With min_gap_size=3, gap=2 is NOT enough to split
        result_no_split = split_trajectories_by_gap(position, min_gap_size=3)
        assert result_no_split.sizes["individuals"] == 1
        assert list(result_no_split.coords["individuals"].values) == [
            "mouse0_000"
        ]

    def test_multiple_individuals(self):
        """Test splitting with multiple individuals.

        Each individual should be processed independently.
        """
        position = _create_test_position_data(
            n_frames=6,
            n_space=2,
            n_keypoints=2,
            n_individuals=2,
            individual_names=["mouse0", "mouse1"],
        )

        # Mouse0: gap at frames 2-3 (will create 2 segments)
        position.loc[{"time": [2, 3], "individuals": "mouse0"}] = np.nan

        # Mouse1: no gap (will create 1 segment)
        # (no modification needed, all frames valid)

        result = split_trajectories_by_gap(position, min_gap_size=1)

        # Should have 3 individuals: mouse0_000, mouse0_001, mouse1_000
        assert result.sizes["individuals"] == 3
        expected_ids = ["mouse0_000", "mouse0_001", "mouse1_000"]
        assert list(result.coords["individuals"].values) == expected_ids

    def test_no_gaps_single_segment(self):
        """Test that data without gaps creates a single segment."""
        position = _create_test_position_data(
            n_frames=6,
            n_space=2,
            n_keypoints=2,
            n_individuals=1,
            individual_names=["mouse0"],
        )

        result = split_trajectories_by_gap(position, min_gap_size=1)

        # Should have 1 individual
        assert result.sizes["individuals"] == 1
        assert list(result.coords["individuals"].values) == ["mouse0_000"]

        # All data should be preserved (non-NaN)
        assert not np.isnan(result.values).any()

    def test_preserves_original_data(self):
        """Test that original DataArray is not modified."""
        position = _create_test_position_data(
            n_frames=6,
            n_space=2,
            n_keypoints=2,
            n_individuals=1,
            individual_names=["mouse0"],
        )

        # Store original values
        original_values = position.values.copy()
        original_individuals = list(position.coords["individuals"].values)

        # Insert NaNs and split
        position.loc[{"time": [2, 3]}] = np.nan

        _ = split_trajectories_by_gap(position, min_gap_size=1)

        # Original should still have NaNs we inserted (but not be further modified)
        assert (
            list(position.coords["individuals"].values) == original_individuals
        )

    def test_preserves_attributes(self):
        """Test that DataArray attributes are preserved."""
        position = _create_test_position_data(
            n_frames=6, n_space=2, n_keypoints=2, n_individuals=1
        )
        position.attrs["custom_attr"] = "test_value"

        position.loc[{"time": [2, 3]}] = np.nan

        result = split_trajectories_by_gap(position, min_gap_size=1)

        assert result.attrs.get("source_software") == "test"
        assert result.attrs.get("fps") == 30
        assert result.attrs.get("custom_attr") == "test_value"

    def test_all_nan_individual(self):
        """Test behavior when an individual has all NaN values."""
        position = _create_test_position_data(
            n_frames=6,
            n_space=2,
            n_keypoints=2,
            n_individuals=2,
            individual_names=["mouse0", "mouse1"],
        )

        # Mouse0: all NaN (should produce no segments)
        position.loc[{"individuals": "mouse0"}] = np.nan

        # Mouse1: valid data
        # (no modification needed)

        result = split_trajectories_by_gap(position, min_gap_size=1)

        # Should only have 1 individual (mouse1_000)
        assert result.sizes["individuals"] == 1
        assert list(result.coords["individuals"].values) == ["mouse1_000"]

    def test_partial_nan_in_keypoints(self):
        """Test that a frame is valid if ANY keypoint is non-NaN.

        Only frames where ALL features (space × keypoints) are NaN
        should be considered invalid.
        """
        position = _create_test_position_data(
            n_frames=6,
            n_space=2,
            n_keypoints=2,
            n_individuals=1,
            individual_names=["mouse0"],
        )

        # Set only ONE keypoint to NaN at frames 2-3
        # Frame should still be considered valid
        position.loc[{"time": [2, 3], "keypoints": "kp0"}] = np.nan

        result = split_trajectories_by_gap(position, min_gap_size=1)

        # Should still be 1 segment (no split because kp1 is still valid)
        assert result.sizes["individuals"] == 1

    def test_invalid_min_gap_size(self):
        """Test that min_gap_size < 1 raises an error."""
        position = _create_test_position_data(
            n_frames=6, n_space=2, n_keypoints=2, n_individuals=1
        )

        with pytest.raises(ValueError, match="min_gap_size must be >= 1"):
            split_trajectories_by_gap(position, min_gap_size=0)

        with pytest.raises(ValueError, match="min_gap_size must be >= 1"):
            split_trajectories_by_gap(position, min_gap_size=-1)

    def test_invalid_dimensions(self):
        """Test that invalid input dimensions raise an error."""
        # Create DataArray with wrong dimensions
        data = np.ones((6, 2, 2))  # Missing individuals dimension
        invalid_position = xr.DataArray(
            data,
            dims=("time", "space", "keypoints"),
            coords={
                "time": np.arange(6),
                "space": ["x", "y"],
                "keypoints": ["kp0", "kp1"],
            },
        )

        with pytest.raises(ValueError):
            split_trajectories_by_gap(invalid_position, min_gap_size=1)

    def test_consecutive_single_frame_gaps(self):
        """Test multiple small gaps that don't merge.

        Time:  0 1 2 3 4 5 6 7 8
        Valid: T F T F T F T F T

        With min_gap_size=1, each single-frame gap should cause a split.
        Expected: 5 segments.
        """
        position = _create_test_position_data(
            n_frames=9,
            n_space=2,
            n_keypoints=2,
            n_individuals=1,
            individual_names=["mouse0"],
        )

        # Insert NaNs at odd frames
        position.loc[{"time": [1, 3, 5, 7]}] = np.nan

        result = split_trajectories_by_gap(position, min_gap_size=1)

        # Should have 5 segments
        assert result.sizes["individuals"] == 5
        expected_ids = [f"mouse0_{i:03d}" for i in range(5)]
        assert list(result.coords["individuals"].values) == expected_ids

    def test_gap_at_start_and_end(self):
        """Test gaps at the beginning and end of trajectory.

        Time:  0 1 2 3 4 5
        Valid: F F T T F F

        Expected: 1 segment at frames 2-3.
        """
        position = _create_test_position_data(
            n_frames=6,
            n_space=2,
            n_keypoints=2,
            n_individuals=1,
            individual_names=["mouse0"],
        )

        # Insert NaNs at start and end
        position.loc[{"time": [0, 1, 4, 5]}] = np.nan

        result = split_trajectories_by_gap(position, min_gap_size=1)

        # Should have 1 segment
        assert result.sizes["individuals"] == 1

        # Valid only at frames 2-3
        seg = result.sel(individuals="mouse0_000")
        assert not np.isnan(seg.sel(time=[2, 3]).values).any()
        assert np.isnan(seg.sel(time=[0, 1, 4, 5]).values).all()

    def test_3d_space(self):
        """Test that function works with 3D spatial data (x, y, z)."""
        position = _create_test_position_data(
            n_frames=6,
            n_space=3,  # 3D space
            n_keypoints=2,
            n_individuals=1,
            individual_names=["mouse0"],
        )

        position.loc[{"time": [2, 3]}] = np.nan

        result = split_trajectories_by_gap(position, min_gap_size=1)

        assert result.sizes["individuals"] == 2
        assert result.sizes["space"] == 3
        assert list(result.coords["space"].values) == ["x", "y", "z"]
