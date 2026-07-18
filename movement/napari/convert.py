"""Conversion functions from ``movement`` datasets to napari layers."""

import numpy as np
import pandas as pd
import xarray as xr

from movement.utils.logging import logger


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
    if "edited" in ds:
        data["edited"] = ds["edited"].values.flatten()
        desired_order.append("edited")

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
        DataFrame with properties (individual, keypoint, time, confidence,
        edited) for use with napari layers.

    See Also
    --------
    napari_layers_to_ds :
        The function carrying out the inverse conversion.

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
    points_as_napari: np.ndarray,
    properties: dict,
    properties_with_nans: pd.DataFrame,
    attrs: dict | None = None,
) -> xr.Dataset:
    """Convert napari Points layer data to a ``movement`` dataset.

    Parameters
    ----------
    points_as_napari
        Live napari Points layer data, shape (N, 3):
        (``frame_idx``, ``y``, ``x``).
        NaN rows are excluded (napari cannot handle NaN coordinates),
        so this may be shorter than the full timeline.
    properties
        Live napari Point properties data. It is in-sync with the
        Points layer data. It is a dictionary with keys
        ``individual``, ``keypoint``, ``time``, ``confidence`` and
        ``edited``, each mapping to a list of values, and each value
        corresponding to a point.
    properties_with_nans:
        Properties DataFrame derived from the original loaded dataset
        including any NaN position data.
    attrs
        Attributes of the original loaded dataset (e.g.
        ``source_software``, ``fps``, ``time_unit`` and
        ``source_file``).

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset derived from the napari Points layer,
        containing pose tracks, confidence scores, and associated metadata.

    Raises
    ------
    ValueError
        If no keypoint or individual has any data left, i.e. all
        points have been removed from the dataset.
    NotImplementedError
        If the napari Points layer data does not represent a pose dataset.

    See Also
    --------
    ds_to_napari_layers :
        The function carrying out the inverse conversion.

    Notes
    -----
    The dataset type is inferred from the presence of ``keypoint`` in
    ``properties``. If present, a poses dataset is returned. Currently,
    bounding box datasets are not supported.

    :func:`ds_to_napari_layers` returns a Tracks array of shape (N, 4) with
    columns (``track_id``, ``frame``, ``y``, ``x``). When loading into
    napari, the ``DataLoader`` widget derives a Points layer from this
    Tracks array by dropping the ``track_id`` column, giving a (N, 3)
    array of (``frame``, ``y``, ``x``). The Points layer is considered
    the "source of truth", as it immediately reflects any manipulation
    of the data done in the napari UI. The function
    :func:`napari_layers_to_ds` therefore relies on the Points layer data
    as one of its inputs, and uses it to reconstruct the corresponding
    dataset.

    :func:`ds_to_napari_layers` preserves NaN values in the output arrays,
    but napari cannot handle NaN coordinates, so the ``DataLoader`` widget
    filters them out upon creation of the napari layers. As a result, when
    reconstructing a dataset via :func:`napari_layers_to_ds`, the input arrays
    will have no NaN (i.e. missing) coordinates.
    This function reconstructs the full dataset by restoring missing points
    using the full coordinate structure from ``properties_with_nans``.

    If a keypoint or individual has no remaining points in any frame,
    it is dropped from the returned dataset rather than kept as an
    all-NaN entry. The ``time`` dimension is never reduced in this
    way: a frame from which every point has been removed is kept,
    with NaN values for position and confidence.

    """
    properties_df = pd.DataFrame.from_dict(
        properties
    )  # live data without nans
    fps = attrs.get("fps") if attrs is not None else None

    if "keypoint" in properties_df.columns:
        # Get full coordinates from the original properties with nan
        time_coords = np.sort(properties_with_nans["time"].unique())
        space_coords = ["x", "y"]
        keypoint_coords = properties_with_nans["keypoint"].unique().tolist()
        individual_coords = (
            properties_with_nans["individual"].unique().tolist()
        )

        # Build position dataframe from napari's live point layer data
        position_df = pd.DataFrame(
            points_as_napari, columns=["frame", "y", "x"]
        )

        # Use the frame coordinate from the live napari layer as the
        # source of truth for time. This avoids relying on
        # properties_df["time"], which may become stale when users add
        # points in napari because new points inherit the properties of
        # the last selected point.
        position_df["time"] = (
            position_df["frame"] / fps if fps else position_df["frame"]
        )

        position_df["keypoint"] = properties_df["keypoint"].to_numpy()
        position_df["individual"] = properties_df["individual"].to_numpy()
        confidence_da = (
            properties_df.set_index(["time", "keypoint", "individual"])[
                "confidence"
            ]
            .to_xarray()
            .reindex(
                time=time_coords,
                keypoint=keypoint_coords,
                individual=individual_coords,
            )
        )
        if "edited" in properties_df.columns:
            edited_da = (
                properties_df.set_index(["time", "keypoint", "individual"])[
                    "edited"
                ]
                .to_xarray()
                .reindex(
                    time=time_coords,
                    keypoint=keypoint_coords,
                    individual=individual_coords,
                    fill_value=False,
                )
            )
        else:
            edited_da = xr.full_like(confidence_da, False, dtype=bool)

        position_df = position_df.melt(
            id_vars=["time", "frame", "keypoint", "individual"],
            value_vars=["x", "y"],
            var_name="space",
            value_name="position",
        )
        position_da = (
            position_df.set_index(["time", "space", "keypoint", "individual"])[
                "position"
            ]
            .to_xarray()
            .reindex(
                time=time_coords,
                space=space_coords,
                keypoint=keypoint_coords,
                individual=individual_coords,
            )
        )

        ds = xr.Dataset(
            data_vars={
                "position": position_da,
                "confidence": confidence_da,
                "edited": edited_da,
            },
            coords={
                "time": time_coords,
                "space": space_coords,
                "keypoint": keypoint_coords,
                "individual": individual_coords,
            },
            attrs=attrs if attrs is not None else {},
        )
        # Drop keypoints/individuals with no data left; never `time`.
        # `edited` is excluded from the check: it's boolean (fill
        # value False), so it's never "null" and would otherwise
        # prevent any keypoint/individual from ever being dropped.
        dropna_subset = ["position", "confidence"]
        ds = ds.dropna(dim="keypoint", how="all", subset=dropna_subset).dropna(
            dim="individual", how="all", subset=dropna_subset
        )
        if ds.sizes["keypoint"] == 0 or ds.sizes["individual"] == 0:
            raise logger.error(
                ValueError(
                    "No points found in the napari layer. "
                    "This happens when all points have been removed."
                )
            )
        return ds

    raise NotImplementedError(
        "Reconstruction of bounding box datasets from napari layers "
        "is not yet implemented."
    )
