"""Save ``movement`` datasets as pynapple-compatible ``.npz`` files."""

from pathlib import Path

import numpy as np
import xarray as xr

from movement.io.save import register_writer
from movement.utils.logging import logger

# Variables required by the poses and bboxes dataset validators
_REQUIRED_VARS: dict[str, tuple[str, ...]] = {
    "poses": ("position", "confidence"),
    "bboxes": ("position", "shape", "confidence"),
}

# Column names for the bbox ``shape`` variables, following pynapple's
# ``nap.from_movement`` naming
_BBOX_SHAPE_COLUMNS: tuple[str, str] = ("width", "height")


def _dim(ds: xr.Dataset, *candidates: str) -> str:
    """Return the first candidate name that is a coordinate of ``ds``.

    Parameters
    ----------
    ds
        The dataset to search for the coordinate.
    candidates
        Candidate coordinate names, in order of preference. Both the
        current singular dimension names (e.g. ``"individual"``) and the
        legacy plural ones (e.g. ``"individuals"``) are accepted.

    Returns
    -------
    str
        The name of the first matching coordinate.

    Raises
    ------
    KeyError
        If none of the candidates is a coordinate of ``ds``.

    """
    for name in candidates:
        if name in ds.coords:
            return name
    raise logger.error(
        KeyError(
            f"None of {candidates} found in dataset coordinates: "
            f"{list(ds.coords)}."
        )
    )


def _build_table(ds: xr.Dataset) -> tuple[np.ndarray, list[str]]:
    """Stack the dataset variables into columns of a single table.

    Parameters
    ----------
    ds
        A validated ``movement`` poses or bounding boxes dataset whose
        time coordinates are in seconds.

    Returns
    -------
    numpy.ndarray
        Data matrix of shape ``(n_time, n_columns)``.
    list of str
        Column names, aligned with the second axis of the data matrix.

    Notes
    -----
    Column names mirror pynapple's ``nap.from_movement`` convention:
    position columns are ``<keypoint>_<space>`` (e.g. ``snout_x``),
    confidence columns are bare keypoint names (e.g. ``snout``), and bbox
    columns are ``x``, ``y``, ``width``, ``height`` and ``confidence``.
    Multi-individual datasets prefix every column with the individual
    name (e.g. ``Alice_snout_x``). Additional per-frame numeric variables
    (e.g. ``head_direction``) are appended as extra columns.

    """
    ds_type = ds.attrs["ds_type"]
    ind_dim = _dim(ds, "individual", "individuals")
    individuals = list(ds.coords[ind_dim].values)
    multi = len(individuals) > 1
    required = _REQUIRED_VARS[ds_type]
    extra_vars = [v for v in ds.data_vars if v not in required]

    blocks: list[np.ndarray] = []
    names: list[str] = []

    if ds_type == "poses":
        kp_dim = _dim(ds, "keypoint", "keypoints")
        keypoints = list(ds.coords[kp_dim].values)
        spaces = list(ds.coords["space"].values)
        position = ds["position"].transpose("time", ind_dim, kp_dim, "space")
        confidence = ds["confidence"].transpose("time", ind_dim, kp_dim)
        for index, individual in enumerate(individuals):
            prefix = f"{individual}_" if multi else ""
            pos_i = position.isel({ind_dim: index}).values
            n_time = pos_i.shape[0]
            names.extend(
                f"{prefix}{kp}_{space}" for kp in keypoints for space in spaces
            )
            blocks.append(pos_i.reshape(n_time, -1))
            conf_i = confidence.isel({ind_dim: index}).values
            names.extend(f"{prefix}{kp}" for kp in keypoints)
            blocks.append(conf_i.reshape(n_time, -1))
    else:  # bboxes: 2D boxes only, as enforced by the dataset validator
        spaces = list(ds.coords["space"].values)
        position = ds["position"].transpose("time", ind_dim, "space")
        shape = ds["shape"].transpose("time", ind_dim, "space")
        confidence = ds["confidence"].transpose("time", ind_dim)
        for index, individual in enumerate(individuals):
            prefix = f"{individual}_" if multi else ""
            pos_i = position.isel({ind_dim: index}).values
            n_time = pos_i.shape[0]
            names.extend(f"{prefix}{space}" for space in spaces)
            blocks.append(pos_i.reshape(n_time, -1))
            shape_i = shape.isel({ind_dim: index}).values
            names.extend(f"{prefix}{col}" for col in _BBOX_SHAPE_COLUMNS)
            blocks.append(shape_i.reshape(n_time, -1))
            conf_i = confidence.isel({ind_dim: index}).values
            names.append(f"{prefix}confidence")
            blocks.append(conf_i.reshape(n_time, -1))

    for var in extra_vars:
        da = ds[var]
        if not np.issubdtype(da.dtype, np.number):
            logger.warning(
                f"Skipping variable {var!r}: pynapple columns must be numeric."
            )
            continue
        dims = set(da.dims)
        if dims == {"time", ind_dim}:
            values = da.transpose("time", ind_dim)
            for index, individual in enumerate(individuals):
                prefix = f"{individual}_" if multi else ""
                names.append(f"{prefix}{var}")
                blocks.append(
                    values.isel({ind_dim: index}).values.reshape(-1, 1)
                )
        elif dims == {"time"}:
            names.append(str(var))
            blocks.append(da.values.reshape(-1, 1))
        else:
            logger.warning(
                f"Skipping variable {var!r}: expected dims (time) or "
                f"(time, {ind_dim}), got {da.dims}."
            )

    return np.concatenate(blocks, axis=1), names


@register_writer("pynapple", suffixes={".npz"})
def to_pynapple_file(ds: xr.Dataset, file: str | Path) -> None:
    """Save a ``movement`` dataset as a pynapple-compatible ``.npz`` file.

    The file stores a single time series table that pynapple loads as a
    ``TsdFrame`` via ``pynapple.load_file``, keeping position,
    confidence and any extra per-frame variables (e.g.
    ``head_direction``) together in one object. Multi-individual
    datasets are flattened into one set of columns, prefixed by
    individual name. Bounding box datasets are supported as well.

    Parameters
    ----------
    ds
        The ``movement`` dataset to save. Its time coordinates must be
        in seconds (i.e. the dataset must have been loaded with ``fps``).
    file
        Path to the ``.npz`` file to write.

    Raises
    ------
    ValueError
        If the dataset's time coordinates are not in seconds (i.e.
        ``ds.attrs["time_unit"]`` is not ``"seconds"``).
    FileNotFoundError, PermissionError, FileExistsError, ValueError
        Invalid or unwritable file path, or a suffix other than
        ``.npz`` (raised by the writer validation).

    See Also
    --------
    pynapple.load_file : Load the resulting file as a pynapple object.
    movement.io.save.save_dataset : Unified saving entry point.

    Notes
    -----
    The npz keys (``t``, ``d``, ``start``, ``end``, ``columns``,
    ``type``) follow pynapple's file format, so no pynapple installation
    is required to write the file. Column naming mirrors pynapple's
    ``nap.from_movement`` convention.

    Examples
    --------
    >>> from movement.io import save_pynapple
    >>> save_pynapple.to_pynapple_file(ds, "/path/to/file.npz")

    Or via the unified entry point:

    >>> from movement.io import save_dataset
    >>> save_dataset(ds, "/path/to/file.npz", target_software="pynapple")

    The file then loads in pynapple as:

    >>> import pynapple as nap
    >>> tsdframe = nap.load_file("/path/to/file.npz")

    """
    if ds.attrs.get("time_unit", "seconds") != "seconds":
        raise logger.error(
            ValueError(
                "Cannot export a dataset with time coordinates in frame "
                "numbers: pynapple expects time in seconds. Reload the "
                "dataset with fps provided (e.g. "
                "load_poses.from_dlc_file(path, fps=30))."
            )
        )

    t = ds.coords["time"].values
    data, names = _build_table(ds)
    np.savez(
        file,
        t=t,
        d=data,
        start=np.atleast_1d(t[0]),
        end=np.atleast_1d(t[-1]),
        columns=np.asarray(names, dtype=str),
        type=np.array(["TsdFrame"], dtype=np.str_),
    )
    logger.info(f"Saved dataset to {file}.")
