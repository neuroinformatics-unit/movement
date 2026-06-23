"""Test suite for the movement.napari.convert module."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from napari.layers.base import ActionType
from pandas.testing import assert_frame_equal

from movement.napari.convert import ds_to_napari_layers, napari_layers_to_ds


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


# -------------------- Valid napari layers test --------------------


def _nan_confidence_at_nan_pos(ds):
    """Return the dataset expected after a napari layer round-trip.

    Points with a NaN position are hidden in napari, so their confidence
    cannot be preserved — the reconstructed dataset has NaN confidence for
    those points.
    """
    position_is_nan = ds["position"].isnull().all("space")
    expected_ds = ds.copy(deep=True)
    expected_ds["confidence"] = xr.where(
        position_is_nan,
        np.nan,
        expected_ds["confidence"],
    )

    return expected_ds


@pytest.mark.parametrize(
    "ds_dataset",
    [
        "dataset",
        "dataset_with_nan",
        "confidence_with_some_nan",
        "confidence_with_all_nan",
    ],
)
def test_valid_poses_roundtrip_napari_layer_to_dataset(ds_dataset, request):
    """Test conversion from napari tracks array to movement pose dataset
    If I convert a dataset to napari and then back to a xarray dataset,
    do I recover the original values? This is a round-trip test.
    """
    ds_name = f"valid_poses_{ds_dataset}"
    ds = request.getfixturevalue(ds_name)
    napari_tracks, _, properties_with_nan = ds_to_napari_layers(ds)

    # simulate loader widget filtering of nans
    valid_point_mask = ~np.any(np.isnan(napari_tracks[:, 2:4]), axis=1)

    # napari_tracks is shape (N,4): (track_id, frame, y, x)
    # but our function is expecting the points layer (N,3): (frame, y, x)
    # the loader widget converts tracks layers to points layer by
    # dropping the track_id column
    napari_points = napari_tracks[valid_point_mask, 1:]
    properties = properties_with_nan.iloc[valid_point_mask].reset_index(
        drop=True
    )

    reconstructed_ds = napari_layers_to_ds(
        napari_points, properties, properties_with_nan, attrs=ds.attrs
    )

    xr.testing.assert_equal(
        reconstructed_ds,
        _nan_confidence_at_nan_pos(ds),
    )


@pytest.mark.parametrize(
    "nan_location",
    [
        None,
        # no NaNs
        {
            "time": "start",
            "individual": ["id_0", "id_1"],
            "keypoint": ["centroid", "left", "right"],
        },  # all individuals are nan at the start
        {
            "time": "end",
            "individual": ["id_0", "id_1"],
            "keypoint": ["centroid", "left", "right"],
        },  # all individuals are nan at the end
        {
            "time": "middle",
            "individual": ["id_0"],
            "keypoint": ["centroid"],
        },  # a single keypoint of an individual is nan mid-sequence
    ],
)
def test_napari_layers_to_ds(
    nan_location,
    valid_poses_path_and_ds,
    valid_poses_path_and_ds_with_localised_nans,
    loaded_data_loader,
):
    # Get sample data filepath and dataset
    if nan_location is None:
        filepath, ds_loaded = valid_poses_path_and_ds
    else:
        filepath, ds_loaded = valid_poses_path_and_ds_with_localised_nans(
            nan_location
        )

    # Get loader widget with sample data loaded
    loader = loaded_data_loader(filepath, ds_loaded)

    # Convert data in napari point layer to a movement dataset
    ds = napari_layers_to_ds(
        points_as_napari=loader.points_layer.data,
        properties=loader.points_layer.properties,  # dict
        properties_with_nans=loader.properties,
        attrs=ds_loaded.attrs,
    )

    xr.testing.assert_equal(ds, _nan_confidence_at_nan_pos(ds_loaded))


def test_napari_layers_to_ds_bboxes_not_implemented():
    """Test bbox reconstruction raises NotImplementedError."""
    with pytest.raises(
        NotImplementedError,
        match="Reconstruction of bounding box datasets",
    ):
        napari_layers_to_ds(
            points_as_napari=np.empty((0, 3)),
            properties={"individual": np.array([])},
            properties_with_nans=pd.DataFrame(),
        )


@pytest.mark.parametrize(
    "nan_location",
    [
        None,
        {
            "time": "start",
            "individual": ["id_0", "id_1"],
            "keypoint": ["centroid", "left", "right"],
        },
        {
            "time": "end",
            "individual": ["id_0", "id_1"],
            "keypoint": ["centroid", "left", "right"],
        },
        {
            "time": "middle",
            "individual": ["id_0"],
            "keypoint": ["centroid"],
        },
    ],
)
def test_edited_pose_napari_layers(
    nan_location,
    valid_poses_path_and_ds,
    valid_poses_path_and_ds_with_localised_nans,
    loaded_data_loader,
):
    """Test that :func:`napari_layers_to_ds` correctly converts edited layers,
    and sets the new confidence value to ``NaN`` after the edit.

    Simulates a user dragging the ``centroid`` of ``id_0`` at frame 2 to new
    ``x``, ``y`` coordinates. Verifies that :func:`napari_layers_to_ds`
    reconstructs a dataset where the edited point has the new position and
    ``NaN`` confidence.
    """
    if nan_location is None:
        filepath, ds_loaded = valid_poses_path_and_ds
    else:
        filepath, ds_loaded = valid_poses_path_and_ds_with_localised_nans(
            nan_location
        )
    loader = loaded_data_loader(filepath, ds_loaded)

    frame = 2  # safe: not NaN in any parametrize case
    keypoint = "centroid"
    individual = "id_0"

    # Find the integer index of the point to edit in the live points layer
    live_props = loader.points_layer.properties
    edit_idx = int(
        np.flatnonzero(
            (live_props["time"] == frame)
            & (live_props["keypoint"] == keypoint)
            & (live_props["individual"] == individual)
        )[0]
    )

    loader.points_layer.data[edit_idx, 1] = 100  # y
    loader.points_layer.data[edit_idx, 2] = 200  # x

    # Direct mutation does not fire napari events, so we call the callback
    # manually below with the correct index to replicate what napari does.
    mock_event = Mock()
    mock_event.source = loader.points_layer
    mock_event.action = ActionType.CHANGED
    mock_event.data_indices = (edit_idx,)
    loader._on_points_data_changed(mock_event)

    ds = napari_layers_to_ds(
        points_as_napari=loader.points_layer.data,
        properties=loader.points_layer.properties,
        properties_with_nans=loader.properties,
        attrs=ds_loaded.attrs,
    )

    # after loading in napari and exporting:
    # confidence is nan where position is nan
    expected_ds = _nan_confidence_at_nan_pos(ds_loaded)
    # edited point values
    expected_ds.position.loc[
        {
            "time": frame,
            "space": ["x", "y"],
            "keypoint": keypoint,
            "individual": individual,
        }
    ] = [200, 100]
    expected_ds["confidence"].loc[
        {
            "time": frame,
            "keypoint": keypoint,
            "individual": individual,
        }
    ] = np.nan
    xr.testing.assert_equal(ds, expected_ds)


@pytest.mark.parametrize(
    "nan_location",
    [
        None,
        {
            "time": "start",
            "individual": ["id_0", "id_1"],
            "keypoint": ["centroid", "left", "right"],
        },
        {
            "time": "end",
            "individual": ["id_0", "id_1"],
            "keypoint": ["centroid", "left", "right"],
        },
        {
            "time": "middle",
            "individual": ["id_0"],
            "keypoint": ["centroid"],
        },
    ],
)
def test_removed_pose_napari_layers(
    nan_location,
    valid_poses_path_and_ds,
    valid_poses_path_and_ds_with_localised_nans,
    loaded_data_loader,
):
    """Test :func:`napari_layers_to_ds` correctly converts a layer with
    removed predictions back to a ``movement`` dataset.

    Simulates a user removing the prediction for ``centroid`` of ``id_0``
    at frame 2 by setting its x, y coordinates and confidence score to NaN.
    Verifies that :func:`napari_layers_to_ds` reconstructs a dataset that
    reflects the removal. The test is parametrized over datasets with and
    without NaN position values to ensure robustness.
    """
    if nan_location is None:
        filepath, ds_expected = valid_poses_path_and_ds
    else:
        filepath, ds_expected = valid_poses_path_and_ds_with_localised_nans(
            nan_location
        )
    loader = loaded_data_loader(filepath, ds_expected)

    frame = 2  # safe: not NaN in any parametrize case
    keypoint = "centroid"
    individual = "id_0"
    remove_mask = (
        (loader.points_layer.properties["time"] == frame)
        & (loader.points_layer.properties["keypoint"] == keypoint)
        & (loader.points_layer.properties["individual"] == individual)
    )
    conf_mask = (
        (loader.properties["time"] == frame)
        & (loader.properties["keypoint"] == keypoint)
        & (loader.properties["individual"] == individual)
    )
    loader.points_layer.data[remove_mask, 1] = np.nan  # y
    loader.points_layer.data[remove_mask, 2] = np.nan  # x
    loader.properties.loc[conf_mask, "confidence"] = np.nan

    ds = napari_layers_to_ds(
        points_as_napari=loader.points_layer.data,
        properties=loader.points_layer.properties,
        properties_with_nans=loader.properties,
        attrs=ds_expected.attrs,
    )
    expected_ds = ds_expected.copy(deep=True)
    expected_ds.position.loc[
        {
            "time": frame,
            "space": ["x", "y"],
            "keypoint": keypoint,
            "individual": individual,
        }
    ] = [np.nan, np.nan]
    expected_ds["confidence"].loc[
        {
            "time": frame,
            "keypoint": keypoint,
            "individual": individual,
        }
    ] = np.nan
    xr.testing.assert_equal(ds, expected_ds)
