"""Conversion functions from ``movement`` datasets to napari layers."""

import numpy as np
import pandas as pd
import xarray as xr

from movement.io import load_poses


def _construct_properties_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Construct a properties DataFrame from a ``movement`` dataset."""
    data = {
        "individual": ds.coords["individual"].values,
        "time": ds.coords["time"].values,
        "confidence": ds["confidence"].values.flatten(),
    }
    desired_order = list(data.keys())
    if "keypoint" in ds.coords:
        data["keypoint"] = ds.coords["keypoint"].values
        desired_order.insert(1, "keypoint")

    # sort
    return pd.DataFrame(data).reindex(columns=desired_order)


def _construct_track_and_time_cols(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute napari track_id and time columns from a ``movement`` dataset."""
    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individual"]
    n_keypoints = ds.sizes.get("keypoint", 1)
    n_tracks = n_individuals * n_keypoints

    # Each keypoint of each individual is a separate track
    track_id_col = np.repeat(np.arange(n_tracks), n_frames).reshape(-1, 1)
    time_col = np.tile(np.arange(n_frames), (n_tracks)).reshape(-1, 1)

    return track_id_col, time_col


def ds_to_napari_layers(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """Convert ``movement`` dataset to napari Tracks array and properties.

    Parameters
    ----------
    ds
        ``movement`` dataset containing pose or bounding box tracks,
        confidence scores, and associated metadata.

    Returns
    -------
    points_as_napari : numpy.ndarray
        position data as a napari Tracks array with shape (N, 4),
        where N is n_keypoints * n_individuals * n_frames
        and the 4 columns are (track_id, frame_idx, y, x).
    bboxes_as_napari : numpy.ndarray | None
        bounding box data as a napari Shapes array with shape (N, 4, 4),
        where N is n_individuals * n_frames and each (4, 4) entry is
        a matrix of 4 rows (1 per corner vertex, starting from upper left
        and progressing in counterclockwise order) with the columns
        (track_id, frame, y, x). Returns None when the input dataset doesn't
        have a "shape" variable.
    properties : pandas.DataFrame
        DataFrame with properties (individual, keypoint, time, confidence)
        for use with napari layers.

    Notes
    -----
    A corresponding napari Points array can be derived from the Tracks array
    by taking its last 3 columns: (frame_idx, y, x). See the documentation
    on the napari Tracks [1]_  and Points [2]_ layers.

    References
    ----------
    .. [1] https://napari.org/stable/howtos/layers/tracks.html
    .. [2] https://napari.org/stable/howtos/layers/points.html

    """
    # Construct the track_ID and time columns for the napari Tracks array
    track_id_col, time_col = _construct_track_and_time_cols(ds)

    # Reorder axes to (individual, keypoint, frames, xy)
    axes_reordering: tuple[int, ...] = (2, 0, 1)
    if "keypoint" in ds.coords:
        axes_reordering = (3,) + axes_reordering
    yx_cols = np.transpose(
        ds.position.values,  # from: frames, xy, keypoint, individual
        axes_reordering,  # to: individual, keypoint, frames, xy
    ).reshape(-1, 2)[:, [1, 0]]  # swap x and y columns

    points_as_napari = np.hstack((track_id_col, time_col, yx_cols))
    bboxes_as_napari = None

    # Construct the napari Shapes array if the input dataset is a
    # bounding boxes one
    if ds.ds_type == "bboxes":
        # Compute bbox corners
        xmin_ymin = ds.position - (ds.shape / 2)
        xmax_ymax = ds.position + (ds.shape / 2)

        # initialise xmax, ymin corner as xmin, ymin
        xmax_ymin = xmin_ymin.copy()
        # overwrite its x coordinate to xmax
        xmax_ymin.loc[{"space": "x"}] = xmax_ymax.loc[{"space": "x"}]

        # initialise xmin, ymin corner as xmin, ymin
        xmin_ymax = xmin_ymin.copy()
        # overwrite its y coordinate to ymax
        xmin_ymax.loc[{"space": "y"}] = xmax_ymax.loc[{"space": "y"}]

        # Add track_id and time columns to each corner array
        corner_arrays_with_track_id_and_time = [
            np.c_[
                track_id_col,
                time_col,
                np.transpose(corner.values, axes_reordering).reshape(-1, 2),
            ]
            for corner in [xmin_ymin, xmin_ymax, xmax_ymax, xmax_ymin]
        ]

        # Concatenate corner arrays along columns
        corners_array = np.concatenate(
            corner_arrays_with_track_id_and_time, axis=1
        )

        # Reshape to napari expected format
        # goes through corners counterclockwise from xmin_ymin
        # in image coordinates
        corners_array = corners_array.reshape(
            -1, 4, 4
        )  # last dimension: track_id, time, x, y
        bboxes_as_napari = corners_array[
            :, :, [0, 1, 3, 2]
        ]  # swap x and y columns

    # Construct the properties DataFrame
    # Stack individual, time and keypoint (if present) dimensions
    # into a new single dimension named "tracks"
    dimensions_to_stack: tuple[str, ...] = ("individual", "time")
    if "keypoint" in ds.coords:
        dimensions_to_stack += ("keypoint",)  # add last
    ds_ = ds.stack(tracks=sorted(dimensions_to_stack))

    properties = _construct_properties_dataframe(ds_)

    return points_as_napari, bboxes_as_napari, properties


def napari_layers_to_ds(
    napari_layers: np.ndarray,
    properties: pd.DataFrame,
    fps: float | None = None,
    valid_point_mask: np.ndarray | None = None,
) -> xr.Dataset:
    """Convert napari layer data back to a ``movement`` dataset.

    Parameters
    ----------
    napari_layers
        Napari layer data generated by
        ``movement.napari.convert.ds_to_napari_layers``.
        This can be either a Points array with shape (N, 3),
        where the 3 columns are (frame_idx, y, x),
        or a Shapes array representing bounding box data
        (rectangle vertices).
    properties
        DataFrame with properties (individual, keypoint, time, confidence)
        for use with napari layers.
    fps
        Frames per second of the video. Defaults to None, in which case
        the ``time`` coordinates will be in frame numbers.
    valid_point_mask
        Boolean mask indicating which rows in ``napari_layers`` are valid.
        Defaults to None, in which case all rows are assumed to be valid.

    Returns
    -------
    ds : xarray.Dataset
        ``movement`` dataset containing pose or bounding box tracks,
        confidence scores, and associated metadata.

    Notes
    -----
    The dataset type is inferred from the presence of ``keypoint`` in
    ``properties``. If present, a poses dataset is returned. Otherwise,
    a bounding boxes dataset is returned.

    ``ds_to_napari_layers`` returns a Tracks array of shape (N, 4) with
    columns (track_id, frame, y, x). When loading into napari,
    ``loader_widgets`` derives a Points layer from this by dropping the
    ``track_id`` column, giving a (N, 3) array of (frame, y, x). The
    Points layer is preferred for editing purposes, as it allows the user
    to directly manipulate individual keypoint positions. This function
    therefore expects the Points layer data as input, and uses it to
    reconstruct the original dataset.

    ``ds_to_napari_layers`` preserves NaN values in the output arrays,
    but napari cannot handle NaN coordinates, so ``loader_widgets`` filters
    them out before passing data to the napari layers. As a result, when
    reconstructing a dataset via ``napari_layers_to_ds``, the input arrays
    may be shorter than the original. Pass ``valid_point_mask`` to indicate
    which rows were kept, so the full array (including NaN slots) can be
    correctly reconstructed.

    """
    if "keypoint" in properties.columns:
        individual_names = properties["individual"].unique().tolist()
        keypoint_names = properties["keypoint"].unique().tolist()

        if valid_point_mask is None:
            valid_point_mask = np.full(len(napari_layers), True)

        position_with_nans = np.full((len(valid_point_mask), 2), np.nan)
        confidence_with_nans = np.full(len(valid_point_mask), np.nan)

        position_with_nans[valid_point_mask] = napari_layers[:, [2, 1]]
        confidence_with_nans[valid_point_mask] = properties[
            "confidence"
        ].to_numpy()

        pose_position = position_with_nans.reshape(
            len(individual_names), len(keypoint_names), -1, 2
        ).transpose(2, 3, 1, 0)
        pose_confidence = confidence_with_nans.reshape(
            len(individual_names), len(keypoint_names), -1
        ).transpose(2, 1, 0)

        return load_poses.from_numpy(
            position_array=pose_position,
            confidence_array=pose_confidence,
            individual_names=individual_names,
            keypoint_names=keypoint_names,
            fps=fps,
        )

    else:  ##bboxes
        raise NotImplementedError(
            "Reconstruction of bounding box datasets from napari layers "
            "is not yet implemented."
        )
