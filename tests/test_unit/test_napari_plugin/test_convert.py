"""Test suite for the movement.napari.convert module."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from movement.napari.convert import poses_to_napari_tracks


@pytest.fixture
def confidence_with_some_nan(valid_poses_dataset_uniform_linear_motion):
    """Return a valid poses dataset with some NaNs in confidence values."""
    ds = valid_poses_dataset_uniform_linear_motion
    ds["confidence"].loc[{"individuals": "id_1", "time": [3, 7, 8]}] = np.nan
    return ds


@pytest.fixture
def confidence_with_all_nan(valid_poses_dataset_uniform_linear_motion):
    """Return a valid poses dataset with all NaNs in confidence values."""
    ds = valid_poses_dataset_uniform_linear_motion
    ds["confidence"].data = np.full_like(ds["confidence"].data, np.nan)
    return ds


@pytest.mark.parametrize(
    "ds_name",
    [
        "valid_poses_dataset_uniform_linear_motion",
        "valid_poses_dataset_uniform_linear_motion_with_nans",
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

    # Prepare expected y, x positions and corresponding confidence values.
    # Assume values are extracted from the dataset in a specific way,
    # by iterating first over individuals and then over keypoints.
    y_coords, x_coords, confidence = [], [], []
    for id in ds.individuals.values:
        for kpt in ds.keypoints.values:
            position = ds.position.sel(individuals=id, keypoints=kpt)
            y_coords.extend(position.sel(space="y").values)
            x_coords.extend(position.sel(space="x").values)
            conf = ds.confidence.sel(individuals=id, keypoints=kpt)
            confidence.extend(conf.values)

    # Generate expected data array
    expected_track_ids = np.repeat(np.arange(n_tracks), n_frames)
    expected_frame_ids = np.tile(np.arange(n_frames), n_tracks)
    expected_yx = np.column_stack((y_coords, x_coords))
    expected_data = np.column_stack(
        (expected_track_ids, expected_frame_ids, expected_yx)
    )

    # Generate expected properties DataFrame
    expected_props = pd.DataFrame(
        {
            "individual": np.repeat(
                ds.individuals.values.repeat(n_keypoints), n_frames
            ),
            "keypoint": np.repeat(
                np.tile(ds.keypoints.values, n_individuals), n_frames
            ),
            "time": expected_frame_ids,
            "confidence": confidence,
        }
    )

    # Assert that the data array matches the expected data
    np.testing.assert_allclose(data, expected_data, equal_nan=True)

    # Assert that the properties DataFrame matches the expected properties
    assert_frame_equal(props, expected_props)


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
