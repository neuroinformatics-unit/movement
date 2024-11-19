"""Test suite for the movement.napari.convert module."""

import numpy as np
import pandas as pd
import pytest

from movement.napari.convert import poses_to_napari_tracks


@pytest.fixture
def confidence_with_some_nan(valid_poses_dataset):
    """Return a valid poses dataset with some NaNs in confidence values."""
    valid_poses_dataset["confidence"].loc[
        {"individuals": "ind1", "time": [3, 7, 8]}
    ] = np.nan
    return valid_poses_dataset


@pytest.fixture
def confidence_with_all_nan(valid_poses_dataset):
    """Return a valid poses dataset with all NaNs in confidence values."""
    valid_poses_dataset["confidence"].data = np.full_like(
        valid_poses_dataset["confidence"].data, np.nan
    )
    return valid_poses_dataset


@pytest.mark.parametrize(
    "ds_name",
    [
        "valid_poses_dataset",
        "valid_poses_dataset_with_nan",
        "confidence_with_some_nan",
        "confidence_with_all_nan",
    ],
)
def test_valid_poses_to_napari_tracks(ds_name, request):
    """Test that the conversion from movement poses dataset to napari
    tracks returns the expected data and properties.
    """
    ds = request.getfixturevalue(ds_name)
    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individuals"]
    n_keypoints = ds.sizes["keypoints"]
    n_tracks = n_individuals * n_keypoints  # total tracked points

    data, props = poses_to_napari_tracks(ds)

    # Check the types and shapes of the returned values
    assert isinstance(data, np.ndarray)
    assert data.shape == (n_frames * n_tracks, 4)
    assert isinstance(props, pd.DataFrame)
    assert len(props) == n_frames * n_tracks

    # Check that the data array has the expected values
    # first column should be each track id repeated for n_frames
    expected_track_ids = np.repeat(np.arange(n_tracks), n_frames)
    np.testing.assert_allclose(data[:, 0], expected_track_ids)
    # 2nd column should be each frame ids repeated for n_tracks
    expected_frame_ids = np.tile(np.arange(n_frames), n_tracks)
    np.testing.assert_allclose(data[:, 1], expected_frame_ids)
    # The last two columns should be the y and x coordinates
    base = np.arange(n_frames, dtype=float)
    expected_yx = np.column_stack(
        (np.tile(base * 4, n_tracks), np.tile(base * base, n_tracks))
    )
    if ds_name == "valid_poses_dataset_with_nan":
        expected_yx[[3, 7, 8, 13, 17, 18]] = np.nan
    np.testing.assert_allclose(data[:, 2:], expected_yx, equal_nan=True)

    # Check that the properties dataframe has the expected values
    expected_columns = ["individual", "keypoint", "time", "confidence"]
    assert all(col in props.columns for col in expected_columns)

    # check that the properties match their expected values
    # note: track_id should be individual_id * n_keypoints + keypoint_id
    expected_individuals = [
        f"ind{i}" for i in expected_track_ids // n_keypoints + 1
    ]
    assert np.array_equal(props["individual"], expected_individuals)
    expected_keypoints = [
        f"key{i}" for i in expected_track_ids % n_keypoints + 1
    ]
    assert np.array_equal(props["keypoint"], expected_keypoints)
    np.testing.assert_allclose(props["time"], expected_frame_ids)


@pytest.mark.parametrize(
    "ds_name, expected_exception",
    [
        ("not_a_dataset", AttributeError),
        ("empty_dataset", KeyError),
        ("missing_var_poses_dataset", AttributeError),
        ("missing_dim_poses_dataset", KeyError),
    ],
)
def test_invalid_poses_to_napari_tracks(ds_name, expected_exception, request):
    """Test that the conversion from movement poses dataset to napari
    tracks raises the expected error for invalid datasets.
    """
    ds = request.getfixturevalue(ds_name)
    with pytest.raises(expected_exception):
        poses_to_napari_tracks(ds)
