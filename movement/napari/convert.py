"""Conversion functions from ``movement`` datasets to napari layers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from movement.utils.logging import logger
from movement.validators.datasets import DS_TYPE_VALIDATORS


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
        edited = ds["edited"].fillna(False).astype(bool)
        data["edited"] = edited.values.flatten()
        desired_order.append("edited")

    # sort
    return pd.DataFrame(data).reindex(columns=desired_order)


def _construct_track_and_time_cols(ds: xr.Dataset) -> np.ndarray:
    """Compute napari track_id and time columns from a ``movement`` dataset.

    Returns an (N, 2) array with columns (track_id, time), where
    N is n_individuals * n_keypoints * n_frames.
    """
    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individual"]
    n_keypoints = ds.sizes.get("keypoint", 1)
    n_tracks = n_individuals * n_keypoints

    # Each keypoint of each individual is a separate track
    track_id_col = np.repeat(np.arange(n_tracks), n_frames)
    time_col = np.tile(np.arange(n_frames), n_tracks)

    return np.column_stack((track_id_col, time_col))


def _validate_ds_for_napari(ds: xr.Dataset) -> None:
    """Check that a ``movement`` dataset can be shown as napari layers.

    The dataset must be 2D, with ``space`` coordinates ``x`` and ``y``.
    The variables used for the napari layers (``position``, ``shape``,
    ``confidence`` and ``edited``) must not have dimensions beyond
    ``time``, ``space``, ``individual`` and (for poses) ``keypoint``.
    Other data variables and dimensions used by them are ignored.

    Raises
    ------
    ValueError
        If the dataset is not 2D or the variables used for the napari
        layers have extra dimensions (e.g. ``view`` in multi-view data).

    """
    space_coords = ds.space.values.tolist()
    if set(space_coords) != {"x", "y"}:
        raise logger.error(
            ValueError(
                "Only 2D data with 'space' coordinates ['x', 'y'] is "
                f"supported, but got {space_coords}."
            )
        )
    validator = DS_TYPE_VALIDATORS[ds.ds_type]
    for var_name in (*validator.VAR_NAMES, "edited"):
        if var_name not in ds:
            continue
        extra_dims = [
            dim for dim in ds[var_name].dims if dim not in validator.DIM_NAMES
        ]
        if extra_dims:
            raise logger.error(
                ValueError(
                    f"Unsupported dimension(s) {extra_dims} in "
                    f"'{var_name}'. Select a single value along each "
                    f"dimension first, e.g. ds.sel({extra_dims[0]}=...)."
                )
            )


def _flatten_to_yx(da: xr.DataArray, row_dims: tuple[str, ...]) -> np.ndarray:
    """Flatten a spatial DataArray into an (N, 2) array of (y, x) coords."""
    return (
        da.sel(space=["y", "x"])
        .transpose(*row_dims, "space")
        .values.reshape(-1, 2)
    )


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
        for use with napari layers. An ``edited`` column is also included
        if the input dataset has an ``edited`` variable.

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
    _validate_ds_for_napari(ds)
    # Construct the track_ID and time columns for the napari Tracks array
    track_id_and_time = _construct_track_and_time_cols(ds)

    # Order rows by (individual, keypoint, time) to match the track_id and
    # time columns
    row_dims: tuple[str, ...] = ("individual", "time")
    if "keypoint" in ds.coords:
        row_dims = ("individual", "keypoint", "time")
    yx_cols = _flatten_to_yx(ds.position, row_dims)

    points_as_napari = np.hstack((track_id_and_time, yx_cols))
    bboxes_as_napari = None

    # Construct the napari Shapes array if the input dataset is a
    # bounding boxes one
    if ds.ds_type == "bboxes":
        # Corners as centre +/- half the bbox size, in (y, x) order
        # and counterclockwise from xmin_ymin in image coordinates:
        # xmin_ymin, xmin_ymax, xmax_ymax, xmax_ymin
        half_size = _flatten_to_yx(ds.shape, row_dims) / 2
        corner_signs = np.array(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=half_size.dtype
        )
        corners = (
            yx_cols[:, np.newaxis] + corner_signs * half_size[:, np.newaxis]
        )  # (N, 4, 2)
        # Repeat the track_id and time columns for each corner
        corner_track_id_and_time = np.broadcast_to(
            track_id_and_time[:, np.newaxis], (len(corners), 4, 2)
        )
        # last dimension: track_id, time, y, x
        bboxes_as_napari = np.concatenate(
            (corner_track_id_and_time, corners), axis=-1
        )

    # Construct the properties DataFrame
    # Stack individual, time and keypoint (if present) dimensions
    # into a new single dimension named "tracks"
    properties = _construct_properties_dataframe(ds.stack(tracks=row_dims))
    if ds.ds_type == "poses" and "keypoint" not in ds.confidence.dims:
        # The loader keeps this full table in the Points layer metadata.
        properties.attrs["individual_wise_confidence"] = True

    properties = _construct_properties_dataframe(ds.stack(tracks=row_dims))

    if ds.ds_type == "poses" and "keypoint" not in ds.confidence.dims:
        # The loader keeps this full table in the Points layer metadata.
        properties.attrs["individual_wise_confidence"] = True

    return points_as_napari, bboxes_as_napari, properties


@dataclass(frozen=True)
class LoadedPointKeys:
    """Integer keys for the loaded points, to match against live napari points.

    Each loaded point's (frame, keypoint, individual) is encoded as an
    int, so matching can be done via array indexing.

    Attributes
    ----------
    keys
        Key of each row of the loaded properties table.
    frames
        Frame index of each row of the loaded properties table.
    id_columns
        ``(column, unique values)`` pairs for the identity columns
        (``keypoint``, ``individual``) present in the loaded table.
    n_keys
        Number of possible keys.

    """

    keys: np.ndarray
    frames: np.ndarray
    id_columns: tuple[tuple[str, pd.Index], ...]
    n_keys: int

    @classmethod
    def from_properties(
        cls, properties_with_nans: pd.DataFrame
    ) -> "LoadedPointKeys":
        """Build keys for every row of the loaded properties table."""
        # Frame index is the time's position along the sorted time axis,
        # matching the frame column built by ds_to_napari_layers
        times = np.sort(properties_with_nans["time"].unique())
        frames = np.searchsorted(times, properties_with_nans["time"])
        keys = frames.astype(np.int64)
        id_columns = []
        n_keys = len(times)
        for col in ("keypoint", "individual"):
            if col in properties_with_nans:
                codes, uniques = pd.factorize(properties_with_nans[col])
                keys = keys * len(uniques) + codes
                id_columns.append((col, pd.Index(uniques)))
                n_keys *= len(uniques)
        return cls(keys, frames, tuple(id_columns), n_keys)

    def present(self, live_frames: np.ndarray, live_properties) -> np.ndarray:
        """Flag the loaded rows that have a matching live point."""
        live_keys = np.asarray(live_frames).astype(np.int64)
        valid = live_keys >= 0
        for col, uniques in self.id_columns:
            codes = uniques.get_indexer(live_properties[col])
            valid &= codes >= 0
            live_keys = live_keys * len(uniques) + codes
        valid &= live_keys < self.n_keys
        present = np.zeros(self.n_keys, dtype=bool)
        present[live_keys[valid]] = True
        return present[self.keys]


def removed_points_mask(
    properties_with_nans: pd.DataFrame,
    live_frames: np.ndarray,
    live_properties,
    keys: LoadedPointKeys | None = None,
) -> np.ndarray:
    """Flag the loaded points that count as removed.

    A point counts as removed if it has no corresponding live napari point and
    was either present at load (removed in this session) or
    flagged as ``edited`` at load (removed in an earlier session).

    Parameters
    ----------
    properties_with_nans
        Properties DataFrame of the loaded dataset, including the NaN
        position rows flagged in its ``position_is_nan`` column.
    live_frames
        Frame index of each live point.
    live_properties
        Live point properties (DataFrame or dict of arrays), with the
        identity columns of ``properties_with_nans``.
    keys
        Precomputed keys for ``properties_with_nans``, to avoid
        rebuilding them on every call. Built here if not given.

    Returns
    -------
    numpy.ndarray
        Boolean mask, row-aligned with ``properties_with_nans``.

    """
    if keys is None:
        keys = LoadedPointKeys.from_properties(properties_with_nans)
    removable = ~properties_with_nans["position_is_nan"].to_numpy()
    if "edited" in properties_with_nans:
        removable |= properties_with_nans["edited"].to_numpy(dtype=bool)
    return removable & ~keys.present(live_frames, live_properties)


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
        ``individual``, ``keypoint``, ``time`` and ``confidence``
        (plus ``edited`` once any point has been dragged), each mapping
        to a list of values, and each value corresponding to a point.
    properties_with_nans:
        Properties DataFrame derived from the original loaded dataset
        including any NaN position data. Must include a
        ``position_is_nan`` boolean column flagging any such points.
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
    properties_df = pd.DataFrame(properties)  # live data without nans
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
        # Points dragged in napari gain an `edited` flag; the column is
        # absent until the first drag.
        if "edited" not in properties_df.columns:
            properties_df["edited"] = False

        # Reconstruct the confidence and edited arrays from the live
        # properties DataFrame
        live_das = (
            properties_df.set_index(["time", "keypoint", "individual"])[
                ["confidence", "edited"]
            ]
            .to_xarray()
            .reindex(
                time=time_coords,
                keypoint=keypoint_coords,
                individual=individual_coords,
            )
        )
        confidence_da = live_das["confidence"]

        # Reconstruct the position array from the live napari Points layer
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

        removed = removed_points_mask(
            properties_with_nans,
            live_frames=points_as_napari[:, 0],
            live_properties=properties_df,
        )
        removed_da = (
            properties_with_nans.assign(removed=removed)
            .set_index(["time", "keypoint", "individual"])["removed"]
            .to_xarray()
            .reindex(
                time=time_coords,
                keypoint=keypoint_coords,
                individual=individual_coords,
                fill_value=False,
            )
        )
        # A point counts as edited if it was dragged in napari or removed
        edited_da = live_das["edited"].fillna(False).astype(bool) | removed_da

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
        # `edited` is excluded from the check: it's boolean (fill value
        # False), so it's never "null" and would otherwise prevent any
        # keypoint/individual from ever being dropped.
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
        if properties_with_nans.attrs.get("individual_wise_confidence"):
            # The loaded table contains the original score even for points
            # with missing positions. Each keypoint has the same copy.
            original = (
                properties_with_nans.drop_duplicates(["time", "individual"])
                .set_index(["time", "individual"])["confidence"]
                .to_xarray()
            )
            # Dragging clears live confidence. A deleted point has no
            # live row. Previously missing points can already carry an
            # edited flag, so only consider positions visible at load.
            visible_at_load = ~properties_with_nans.set_index(
                ["time", "keypoint", "individual"]
            )["position_is_nan"].to_xarray()
            changed = visible_at_load & edited_da & confidence_da.isnull()
            ds["confidence"] = original.where(~changed.any("keypoint"))

        # Drop `edited` entirely if nothing survived the trim above:
        # e.g. the only edits were on a keypoint/individual just dropped.
        if not ds["edited"].any():
            ds = ds.drop_vars("edited")
        return ds

    raise NotImplementedError(
        "Reconstruction of bounding box datasets from napari layers "
        "is not yet implemented."
    )
