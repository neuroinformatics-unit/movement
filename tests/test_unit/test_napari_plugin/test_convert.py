"""Test suite for the movement.napari.convert module."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from movement.napari.convert import ds_to_napari_layers


def set_some_confidence_values_to_nan(ds, individuals, time):
    """Set some confidence values to NaN for specific individuals and time."""
    ds["confidence"].loc[{"individual": individuals, "time": time}] = np.nan
    return ds


def set_all_confidence_values_to_nan(ds):
    """Set all confidence values to NaN."""
    ds["confidence"].data = np.full_like(ds["confidence"].data, np.nan)
    return ds


def _get_napari_array_for_poses(y_coords, x_coords, n_tracks, n_frames):
    """Return input data as a napari Tracks array."""
    track_ids = np.repeat(np.arange(n_tracks), n_frames)
    frame_ids = np.tile(np.arange(n_frames), n_tracks)
    yx = np.column_stack((y_coords, x_coords))

    return np.column_stack((track_ids, frame_ids, yx))


def _get_napari_arrays_for_bboxes(
    y_coords, x_coords, heights, widths, n_tracks, n_frames
):
    """Return input data as napari Tracks and Shapes arrays."""
    # Generate napari tracks array from input data
    napari_tracks_array = _get_napari_array_for_poses(
        y_coords, x_coords, n_tracks, n_frames
    )
    track_ids = napari_tracks_array[:, 0]
    frame_ids = napari_tracks_array[:, 1]
    yx = napari_tracks_array[:, -2:]

    hw = np.column_stack((heights, widths))

    # Compute corner position arrays
    xmin_ymin = np.flip(yx - (hw / 2), axis=1)
    xmax_ymax = np.flip(yx + (hw / 2), axis=1)

    xmin_ymax = xmin_ymin.copy()
    xmin_ymax[:, 1] = xmax_ymax[:, 1]
    xmax_ymin = xmin_ymin.copy()
    xmax_ymin[:, 0] = xmax_ymax[:, 0]

    # Add expected time/track columns
    corner_arrays_with_track_id_and_time = [
        np.c_[
            track_ids,
            frame_ids,
            corner,
        ]
        for corner in [xmin_ymin, xmin_ymax, xmax_ymax, xmax_ymin]
    ]

    # Concatenate corner arrays along columns
    corners_array = np.concatenate(
        corner_arrays_with_track_id_and_time, axis=1
    )
    # reshape to correct format and order of vertices
    corners_array = corners_array.reshape(
        -1, 4, 4
    )  # last dimension: track_id, time, x, y
    napari_bboxes_array = corners_array[
        :, :, [0, 1, 3, 2]
    ]  # swap x and y columns

    return napari_tracks_array, napari_bboxes_array


@pytest.fixture
def valid_poses_confidence_with_some_nan(valid_poses_dataset):
    """Return a valid poses dataset with some NaNs in confidence values."""
    return set_some_confidence_values_to_nan(
        valid_poses_dataset, individuals=["id_1"], time=[3, 7, 8]
    )


@pytest.fixture
def valid_poses_confidence_with_all_nan(valid_poses_dataset):
    """Return a valid poses dataset with all NaNs in confidence values."""
    return set_all_confidence_values_to_nan(valid_poses_dataset)


@pytest.fixture
def valid_bboxes_confidence_with_some_nan(valid_bboxes_dataset):
    """Return a valid bboxes dataset with some NaNs in confidence values."""
    return set_some_confidence_values_to_nan(
        valid_bboxes_dataset, individuals=["id_1"], time=[3, 7, 8]
    )


@pytest.fixture
def valid_bboxes_confidence_with_all_nan(valid_bboxes_dataset):
    """Return a valid bboxes dataset with all NaNs in confidence values."""
    return set_all_confidence_values_to_nan(valid_bboxes_dataset)


@pytest.mark.parametrize(
    "ds_dataset",
    [
        "dataset",
        "dataset_with_nan",
        "confidence_with_some_nan",
        "confidence_with_all_nan",
    ],
)
def test_valid_poses_dataset_to_napari_arrays(ds_dataset, request):
    """Test that the conversion from movement dataset to napari
    tracks returns the expected data and properties.
    """
    # Combine parametrized inputs into the name of the fixture to test
    ds_name = f"valid_poses_{ds_dataset}"
    ds = request.getfixturevalue(ds_name)

    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individual"]
    n_keypoints = ds.sizes.get("keypoint", 1)
    n_tracks = n_individuals * n_keypoints  # total tracked points

    # Convert the dataset to a napari Tracks array, napari Shapes array
    # (will be None for a poses datasets) and properties dataframe
    napari_tracks, napari_bboxes, properties_df = ds_to_napari_layers(ds)

    # Iterate over individuals and keypoints to extract positions and
    # confidence values.
    # We assume values are extracted from the dataset in a specific way,
    # by iterating first over individuals and then over keypoints.
    y_coords, x_coords, confidence = [], [], []
    for id in ds.individual.values:
        positions = ds.position.sel(individual=id)
        confidences = ds.confidence.sel(individual=id)

        for kpt in ds.keypoint.values:
            y_coords.extend(positions.sel(keypoint=kpt, space="y").values)
            x_coords.extend(positions.sel(keypoint=kpt, space="x").values)
            confidence.extend(confidences.sel(keypoint=kpt).values)

    # Generate expected napari tracks array
    expected_tracks = _get_napari_array_for_poses(
        y_coords, x_coords, n_tracks, n_frames
    )
    expected_frame_ids = expected_tracks[:, 1]

    # Generate expected properties DataFrame
    expected_properties_df = pd.DataFrame(
        {
            "individual": np.repeat(
                ds.individual.values.repeat(n_keypoints), n_frames
            ),
            **(
                {
                    "keypoint": np.repeat(
                        np.tile(ds.keypoint.values, n_individuals), n_frames
                    )
                }
            ),
            "time": np.int64(expected_frame_ids),
            "confidence": confidence,
        }
    )

    # Assert that the napari tracks array matches the expected one
    np.testing.assert_allclose(napari_tracks, expected_tracks, equal_nan=True)
    assert napari_bboxes is None

    # Assert that the properties DataFrame matches the expected one
    assert_frame_equal(properties_df, expected_properties_df)


@pytest.mark.parametrize(
    "ds_dataset",
    [
        "dataset",
        "dataset_with_nan",
        "confidence_with_some_nan",
        "confidence_with_all_nan",
    ],
)
def test_valid_bboxes_dataset_to_napari_arrays(ds_dataset, request):
    """Test that the conversion from movement dataset to napari
    tracks returns the expected data and properties.
    """
    # Combine parametrized inputs into the name of the fixture to test
    ds_name = f"valid_bboxes_{ds_dataset}"
    ds = request.getfixturevalue(ds_name)

    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individual"]
    n_keypoints = ds.sizes.get("keypoint", 1)
    n_tracks = n_individuals * n_keypoints  # total tracked points

    # Convert the dataset to a napari Tracks array, napari Shapes array
    # and properties dataframe
    napari_tracks, napari_bboxes, properties_df = ds_to_napari_layers(ds)

    # Iterate over individuals and keypoints to extract positions and
    # confidence, height and width values.
    # We assume values are extracted from the dataset
    # by iterating first over individuals and then over keypoints.
    y_coords, x_coords, confidence, heights, widths = [], [], [], [], []
    for id in ds.individual.values:
        positions = ds.position.sel(individual=id)
        confidences = ds.confidence.sel(individual=id)
        shapes = ds.shape.sel(individual=id)

        y_coords.extend(positions.sel(space="y").values)
        x_coords.extend(positions.sel(space="x").values)
        confidence.extend(confidences.values)
        heights.extend(shapes.sel(space="y").values)
        widths.extend(shapes.sel(space="x").values)

    # Generate expected napari tracks and bboxes arrays
    expected_tracks, expected_bboxes = _get_napari_arrays_for_bboxes(
        y_coords, x_coords, heights, widths, n_tracks, n_frames
    )

    # Generate expected properties DataFrame
    expected_properties_df = pd.DataFrame(
        {
            "individual": np.repeat(
                ds.individual.values.repeat(n_keypoints), n_frames
            ),
            "time": np.tile(np.arange(n_frames), n_tracks),
            "confidence": confidence,
        }
    )

    # Assert that the napari arrays match the expected ones
    np.testing.assert_allclose(napari_tracks, expected_tracks, equal_nan=True)
    np.testing.assert_allclose(napari_bboxes, expected_bboxes, equal_nan=True)

    # Assert that the properties DataFrame matches the expected one
    assert_frame_equal(
        properties_df,
        expected_properties_df,
    )


@pytest.mark.parametrize(
    "ds_name, expected_exception",
    [
        ("not_a_dataset", AttributeError),
        ("empty_dataset", KeyError),
        ("missing_var_poses_dataset", AttributeError),
        ("missing_dim_poses_dataset", KeyError),
    ],
)
def test_invalid_poses_to_napari_layers(ds_name, expected_exception, request):
    """Test that the conversion from movement poses dataset to napari
    arrays raises the expected error for invalid datasets.
    """
    ds = request.getfixturevalue(ds_name)
    with pytest.raises(expected_exception):
        ds_to_napari_layers(ds)
