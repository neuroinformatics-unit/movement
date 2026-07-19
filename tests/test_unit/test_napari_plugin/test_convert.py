"""Test suite for the movement.napari.convert module."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from napari.layers.base import ActionType
from pandas.testing import assert_frame_equal

from movement.io import save_poses
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


def _move_point(loader, frame, keypoint, individual, new_y, new_x):
    """Simulate a user dragging a single point to a new position."""
    live_props = loader.points_layer.properties
    edit_idx = int(
        np.flatnonzero(
            (live_props["time"] == frame)
            & (live_props["keypoint"] == keypoint)
            & (live_props["individual"] == individual)
        )[0]
    )
    loader.points_layer.data[edit_idx, 1] = new_y
    loader.points_layer.data[edit_idx, 2] = new_x

    mock_event = Mock()
    mock_event.source = loader.points_layer
    mock_event.action = ActionType.CHANGED
    mock_event.data_indices = (edit_idx,)
    loader._on_points_data_changed(mock_event)


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
    _move_point(
        loader,
        frame=frame,
        keypoint=keypoint,
        individual=individual,
        new_y=100,
        new_x=200,
    )

    ds = napari_layers_to_ds(
        points_as_napari=loader.points_layer.data,
        properties=loader.points_layer.properties,
        properties_with_nans=loader.properties,
        attrs=ds_loaded.attrs,
    )
    # check if `edited` is boolean
    assert ds["edited"].dtype == bool

    # after loading in napari and exporting:
    # confidence is nan where position is nan
    expected_ds = _nan_confidence_at_nan_pos(ds_loaded)
    edited_point = {
        "time": frame,
        "keypoint": keypoint,
        "individual": individual,
    }
    expected_ds.position.loc[{**edited_point, "space": ["x", "y"]}] = [
        200,
        100,
    ]
    expected_ds["confidence"].loc[edited_point] = np.nan
    expected_ds["edited"] = xr.full_like(
        expected_ds["confidence"], False, dtype=bool
    )
    expected_ds["edited"].loc[edited_point] = True
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
def test_edited_property_round_trip(
    nan_location,
    valid_poses_path_and_ds,
    valid_poses_path_and_ds_with_localised_nans,
    loaded_data_loader,
):
    """Test that an ``edited`` variable survives a full round trip.

    Once a dataset has been reconstructed with an ``edited`` variable
    (e.g. from a previous napari editing session), it must be possible
    to load it back into napari via :func:`ds_to_napari_layers`, and
    to convert it back again via :func:`napari_layers_to_ds`, without
    losing the ``edited`` flags. This simulates reopening a
    previously-edited dataset in a new napari session.
    """
    if nan_location is None:
        filepath, ds_loaded = valid_poses_path_and_ds
    else:
        filepath, ds_loaded = valid_poses_path_and_ds_with_localised_nans(
            nan_location
        )
    loader = loaded_data_loader(filepath, ds_loaded)

    _move_point(
        loader,
        frame=2,  # safe: not NaN in any parametrize case
        keypoint="centroid",
        individual="id_0",
        new_y=100,
        new_x=200,
    )

    ds = napari_layers_to_ds(
        points_as_napari=loader.points_layer.data,
        properties=loader.points_layer.properties,
        properties_with_nans=loader.properties,
        attrs=ds_loaded.attrs,
    )
    assert "edited" in ds

    # Load the previously-edited dataset back into napari layers.
    napari_tracks, _, properties_with_nan = ds_to_napari_layers(ds)
    assert "edited" in properties_with_nan.columns

    # Simulate the loader widget filtering out NaN position rows, then
    # convert back to a dataset again.
    valid_point_mask = ~np.any(np.isnan(napari_tracks[:, 2:4]), axis=1)
    napari_points = napari_tracks[valid_point_mask, 1:]
    properties = properties_with_nan.iloc[valid_point_mask].reset_index(
        drop=True
    )

    reconstructed_ds = napari_layers_to_ds(
        napari_points, properties, properties_with_nan, attrs=ds.attrs
    )

    assert reconstructed_ds["edited"].dtype == bool
    xr.testing.assert_equal(reconstructed_ds, ds)


def _target_points_to_remove(target, ds):
    """Expand the ``"all"`` shorthand in a removal target into the
    dataset's actual coordinate values for ``time``/``individual``/
    ``keypoint``, so the result can be used to build a removal mask
    (see :func:`_remove_points`) or an xarray ``.loc`` selection.
    """
    return {
        key: (ds.coords[key].values if target[key] == "all" else target[key])
        for key in ("time", "individual", "keypoint")
    }


def _remove_points(loader, target):
    """Simulate napari deleting every point matching ``target`` (as
    produced by :func:`_target_points_to_remove`) from the live Points
    layer, dropping the matching rows from both ``points_layer.data``
    and ``points_layer.properties`` and returning the shrunk arrays.
    """
    live_props = loader.points_layer.properties
    remove_mask = (
        np.isin(live_props["time"], target["time"])
        & np.isin(live_props["individual"], target["individual"])
        & np.isin(live_props["keypoint"], target["keypoint"])
    )
    keep_mask = ~remove_mask
    points = loader.points_layer.data[keep_mask]
    properties = {key: value[keep_mask] for key, value in live_props.items()}
    return points, properties


@pytest.mark.parametrize(
    "nan_location",
    [
        None,
        {"time": "middle", "individual": ["id_0"], "keypoint": ["centroid"]},
    ],
    ids=["no_pre_existing_nans", "pre_existing_nans"],
)
@pytest.mark.parametrize(
    "removal_selector",
    [
        {"time": [2], "individual": ["id_0"], "keypoint": ["centroid"]},
        {"time": "all", "individual": ["id_0"], "keypoint": ["centroid"]},
        {"time": [3], "individual": "all", "keypoint": "all"},
        {"time": [3, 4, 5], "individual": ["id_0"], "keypoint": "all"},
        {"time": [3, 4, 5], "individual": "all", "keypoint": "all"},
    ],
    ids=[
        "keypoint_removed_for_one_individual_one_frame",
        "keypoint_removed_for_one_individual_all_frames",
        "single_frame_removed_all_keypoints_all_individuals",
        "frame_range_removed_one_individual_all_keypoints",
        "frame_range_removed_all_individuals_all_keypoints",
    ],
)
def test_removed_pose_napari_layers_restores_nans(
    nan_location,
    removal_selector,
    valid_poses_path_and_ds,
    valid_poses_path_and_ds_with_localised_nans,
    loaded_data_loader,
):
    """Test partial removal of napari points across a keypoint,
    individual, or frame range.

    Simulates a user selecting and deleting many points at once in the
    napari Points layer: every point for a keypoint (for one individual,
    or for all of them), every point for an individual, or every point
    for a range of frames (for one individual, or for all of them).
    Parametrized over datasets with and without pre-existing NaN
    positions, to check removal still works when combined with data
    that's already missing.
    """
    if nan_location is None:
        filepath, ds_expected = valid_poses_path_and_ds
    else:
        filepath, ds_expected = valid_poses_path_and_ds_with_localised_nans(
            nan_location
        )
    loader = loaded_data_loader(filepath, ds_expected)
    target = _target_points_to_remove(removal_selector, ds_expected)
    points, properties = _remove_points(loader, target)

    ds = napari_layers_to_ds(
        points_as_napari=points,
        properties=properties,
        properties_with_nans=loader.properties,
        attrs=ds_expected.attrs,
    )
    # No point was ever dragged, so the live properties never gained an
    # `edited` key: deletion alone does not add the variable.
    assert "edited" not in ds

    # Removed points are restored as NaN; nothing is dropped from the
    # dataset's time/keypoint/individual coordinates.
    expected_ds = _nan_confidence_at_nan_pos(ds_expected)
    expected_ds.position.loc[
        {
            "time": target["time"],
            "space": ["x", "y"],
            "keypoint": target["keypoint"],
            "individual": target["individual"],
        }
    ] = np.nan
    expected_ds["confidence"].loc[
        {
            "time": target["time"],
            "keypoint": target["keypoint"],
            "individual": target["individual"],
        }
    ] = np.nan

    xr.testing.assert_equal(ds, expected_ds)


@pytest.mark.parametrize(
    "removal_selector, dropped_dim",
    [
        (
            {"time": "all", "individual": "all", "keypoint": ["centroid"]},
            "keypoint",
        ),
        (
            {"time": "all", "individual": ["id_0"], "keypoint": "all"},
            "individual",
        ),
    ],
    ids=[
        "keypoint_removed_for_all_individuals_all_frames",
        "individual_removed_all_keypoints_all_frames",
    ],
)
def test_removed_pose_napari_layers_drops_empty_coord(
    removal_selector,
    dropped_dim,
    valid_poses_path_and_ds,
    loaded_data_loader,
):
    """Test that a keypoint/individual with no data left anywhere is
    dropped entirely from the reconstructed dataset.

    Unlike a partial removal (some, but not all, individuals/frames for
    a keypoint), removing a keypoint or individual entirely leaves it
    with no data anywhere -- ``napari_layers_to_ds`` drops it from the
    dataset rather than keeping it as an all-NaN coordinate.
    """
    filepath, ds_expected = valid_poses_path_and_ds
    loader = loaded_data_loader(filepath, ds_expected)
    target = _target_points_to_remove(removal_selector, ds_expected)
    points, properties = _remove_points(loader, target)

    ds = napari_layers_to_ds(
        points_as_napari=points,
        properties=properties,
        properties_with_nans=loader.properties,
        attrs=ds_expected.attrs,
    )
    # No point was ever dragged, so the live properties never gained an
    # `edited` key: deletion alone does not add the variable.
    assert "edited" not in ds

    removed_labels = target[dropped_dim]
    remaining_labels = [
        label
        for label in ds_expected.coords[dropped_dim].values
        if label not in removed_labels
    ]
    assert list(ds.coords[dropped_dim].values) == remaining_labels

    expected_ds = ds_expected.sel({dropped_dim: remaining_labels})
    xr.testing.assert_equal(ds, expected_ds)


@pytest.mark.parametrize(
    "valid_poses_dataset",
    ["multi_individual_array", "single_individual_array"],
    indirect=True,
    ids=["two_individual_dataset", "single_individual_dataset"],
)
def test_removed_pose_napari_layers_empty_ds(
    valid_poses_dataset,
    tmp_path,
    loaded_data_loader,
):
    """Test that removing every point in the dataset raises a ValueError.

    Covers deleting all individuals (and therefore every point) from a
    2-individual dataset, and deleting the only individual from a
    1-individual dataset. Either way, nothing is left to reconstruct, so
    :func:`napari_layers_to_ds` should raise rather than silently return
    an empty dataset.
    """
    ds_expected = valid_poses_dataset
    filepath = tmp_path / "ds.csv"
    save_poses.to_dlc_file(ds_expected, filepath, split_individuals=False)
    loader = loaded_data_loader(filepath, ds_expected)
    target = _target_points_to_remove(
        {"time": "all", "individual": "all", "keypoint": "all"}, ds_expected
    )
    points, properties = _remove_points(loader, target)

    with pytest.raises(
        ValueError, match="No points found in the napari layer"
    ):
        napari_layers_to_ds(
            points_as_napari=points,
            properties=properties,
            properties_with_nans=loader.properties,
            attrs=ds_expected.attrs,
        )


def test_never_detected_keypoint_dropped_on_round_trip(
    valid_poses_dataset, tmp_path, loaded_data_loader
):
    """Test that a keypoint the tracker never detected is dropped.

    A keypoint that is all-NaN in the source file is filtered out when
    the napari layers are created, so it is indistinguishable from one
    the user deleted: both are simply absent from the live Points layer.
    It is therefore dropped on conversion back to a dataset, even though
    the user edited nothing. This test documents that behaviour.
    """
    ds_in = valid_poses_dataset.copy(deep=True)
    ds_in.position.loc[{"keypoint": "left"}] = np.nan  # never detected
    filepath = tmp_path / "ds.csv"
    save_poses.to_dlc_file(ds_in, filepath, split_individuals=False)
    loader = loaded_data_loader(filepath, ds_in)

    # The keypoint survives loading; it is lost only on conversion back.
    assert "left" in loader.properties["keypoint"].unique()

    # No edits: hand the live layers straight back.
    ds_out = napari_layers_to_ds(
        points_as_napari=loader.points_layer.data,
        properties=loader.points_layer.properties,
        properties_with_nans=loader.properties,
        attrs=ds_in.attrs,
    )

    assert "left" not in ds_out.coords["keypoint"]
    xr.testing.assert_equal(ds_out, ds_in.drop_sel(keypoint="left"))
