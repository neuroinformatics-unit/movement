"""Test suite for the movement.napari.convert module."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from movement.napari.convert import ds_to_napari_layers


def set_some_confidence_values_to_nan(ds, individuals, time):
    """Set some confidence values to NaN for specific individuals and time."""
    ds["confidence"].loc[{"individuals": individuals, "time": time}] = np.nan
    return ds


def set_all_confidence_values_to_nan(ds):
    """Set all confidence values to NaN."""
    ds["confidence"].data = np.full_like(ds["confidence"].data, np.nan)
    return ds


def _get_expected_bboxes_napari(
    expected_track_ids,
    expected_frame_ids,
    expected_yx,
    expected_hw,
):
    # Compute corner position arrays
    xmin_ymin = np.flip(expected_yx - (expected_hw / 2), axis=1)
    xmax_ymax = np.flip(expected_yx + (expected_hw / 2), axis=1)

    xmin_ymax = xmin_ymin.copy()
    xmin_ymax[:, 1] = xmax_ymax[:, 1]
    xmax_ymin = xmin_ymin.copy()
    xmax_ymin[:, 0] = xmax_ymax[:, 0]

    # Add expected time/track columns
    corner_arrays_with_track_id_and_time = [
        np.c_[
            expected_track_ids,
            expected_frame_ids,
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
    expected_bboxes = corners_array[:, :, [0, 1, 3, 2]]  # swap x and y columns
    return expected_bboxes


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
    "ds_data_type",
    ["poses", "bboxes"],
)
@pytest.mark.parametrize(
    "ds_dataset",
    [
        "dataset",
        "dataset_with_nan",
        "confidence_with_some_nan",
        "confidence_with_all_nan",
    ],
)
def test_valid_dataset_to_napari_arrays(ds_data_type, ds_dataset, request):
    """Test that the conversion from movement dataset to napari
    tracks returns the expected data and properties.
    """
    # Combine parametrized inputs into the name of the fixture to test
    ds_name = f"valid_{ds_data_type}_{ds_dataset}"
    ds = request.getfixturevalue(ds_name)

    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individuals"]
    n_keypoints = ds.sizes.get("keypoints", 1)
    n_tracks = n_individuals * n_keypoints  # total tracked points

    # Convert the dataset to a napari Tracks array, napari Shapes array
    # (will be None for non-bboxes datasets) and properties dataframe
    data, bboxes, props = ds_to_napari_layers(ds)

    # Prepare expected y, x positions and corresponding confidence values.
    # Assume values are extracted from the dataset in a specific way,
    # by iterating first over individuals and then over keypoints.
    y_coords, x_coords, confidence = [], [], []
    # Prepare height, width of bounding boxes as well if the dataset is
    # a bounding boxes dataset
    if ds_data_type == "bboxes":
        heights, widths = [], []
    for id in ds.individuals.values:
        positions = ds.position.sel(individuals=id)
        confidences = ds.confidence.sel(individuals=id)
        if ds_data_type == "bboxes":
            shapes = ds.shape.sel(individuals=id)

        if "keypoints" in ds:
            for kpt in ds.keypoints.values:
                y_coords.extend(positions.sel(keypoints=kpt, space="y").values)
                x_coords.extend(positions.sel(keypoints=kpt, space="x").values)
                confidence.extend(confidences.sel(keypoints=kpt).values)
        else:
            y_coords.extend(positions.sel(space="y").values)
            x_coords.extend(positions.sel(space="x").values)
            confidence.extend(confidences.values)
        if ds_data_type == "bboxes":
            heights.extend(shapes.sel(space="y").values)
            widths.extend(shapes.sel(space="x").values)

    # Generate expected data array
    expected_track_ids = np.repeat(np.arange(n_tracks), n_frames)
    expected_frame_ids = np.tile(np.arange(n_frames), n_tracks)
    expected_yx = np.column_stack((y_coords, x_coords))
    expected_data = np.column_stack(
        (expected_track_ids, expected_frame_ids, expected_yx)
    )

    # Generate expected bboxes
    if ds_data_type == "bboxes":
        expected_hw = np.column_stack((heights, widths))
        expected_bboxes = _get_expected_bboxes_napari(
            expected_track_ids,
            expected_frame_ids,
            expected_yx,
            expected_hw,
        )
    else:
        expected_bboxes = None

    # Generate expected properties DataFrame
    expected_props_dict = {
        "individual": np.repeat(
            ds.individuals.values.repeat(n_keypoints), n_frames
        ),
        **(
            {
                "keypoint": np.repeat(
                    np.tile(ds.keypoints.values, n_individuals), n_frames
                )
            }
            if "keypoints" in ds
            else {}
        ),
        "time": expected_frame_ids,
        "confidence": confidence,
    }
    expected_props = pd.DataFrame(expected_props_dict)

    # Assert that the data array matches the expected data
    np.testing.assert_allclose(data, expected_data, equal_nan=True)
    if ds_data_type == "bboxes":
        np.testing.assert_allclose(bboxes, expected_bboxes, equal_nan=True)
    else:
        assert bboxes is None

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
def test_invalid_poses_to_napari_layers(ds_name, expected_exception, request):
    """Test that the conversion from movement poses dataset to napari
    tracks raises the expected error for invalid datasets.
    """
    ds = request.getfixturevalue(ds_name)
    with pytest.raises(expected_exception):
        ds_to_napari_layers(ds)
