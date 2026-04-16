"""Load pose tracking data from aniframe Parquet files into ``movement``."""

import io
import math
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rdata
import xarray as xr

from movement.io.load import register_loader
from movement.io.load_poses import from_numpy
from movement.utils.logging import logger
from movement.validators.files import ValidAniframeParquet, ValidFile

# Recognised column-name sets (mirrors aniframe's own detection heuristics)
_WHAT_COLS: frozenset[str] = frozenset(
    {"model", "individual", "track", "keypoint"}
)
_WHEN_COLS: frozenset[str] = frozenset({"session", "trial", "time"})
_WHERE_CARTESIAN: frozenset[str] = frozenset({"x", "y", "z"})
_WHERE_POLAR: frozenset[str] = frozenset({"rho", "phi", "theta"})

# Conversion factors from each time unit to seconds
_TIME_UNIT_TO_SECONDS: dict[str, float] = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


@register_loader("aniframe", file_validators=ValidAniframeParquet)
def from_aniframe_file(
    file: str | Path, fps: float | None = None
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from an aniframe Parquet file.

    `aniframe <https://animovement.dev/aniframe/>`_ is a long-format tidy
    data structure produced by the
    `animovement <https://animovement.dev/>`_ R ecosystem. Each row
    represents a single observation for one individual, keypoint, and
    timepoint.

    Parameters
    ----------
    file
        Path to the aniframe Parquet file (``.parquet``).
    fps
        Frames per second. If ``None`` (default), the value is read from
        the file's embedded metadata (``sampling_rate`` field). An
        explicitly provided value takes precedence over the metadata.
        If no fps can be determined and ``unit_time`` is not ``"frame"``,
        the time coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Raises
    ------
    ValueError
        If the file contains polar or spherical coordinate columns
        (``rho``, ``phi``, ``theta``), or if any extra identity or temporal
        column (e.g. ``model``, ``session``, ``trial``) holds more than one
        unique value.

    Warns
    -----
    UserWarning
        If the column ``track`` is renamed to ``individual``, or if
        ``point_of_reference`` is ``"bottom_left"`` (which differs from
        the top-left origin used by ``movement`` and napari).

    See Also
    --------
    movement.io.load_poses.from_numpy

    Examples
    --------
    >>> from movement.io import load_dataset
    >>> ds = load_dataset("path/to/file.parquet", source_software="aniframe")

    Notes
    -----
    Only 2D and 3D Cartesian coordinate systems are supported. Extra
    ``variables_what`` or ``variables_when`` columns (e.g., ``model``,
    ``session``, ``trial``) are silently dropped when they hold a single
    unique value; they raise a ``ValueError`` when they hold multiple values.

    The original tracking software is taken from the ``source`` field in
    the aniframe metadata (e.g., ``"SLEAP"`` or ``"DeepLabCut"``).

    """
    valid_file = cast("ValidFile", file)
    file_path = valid_file.file

    df = pd.read_parquet(file_path)
    meta = _decode_aniframe_metadata(file_path)
    df = _resolve_columns(df)
    _check_no_polar_coords(set(df.columns))

    # Detect which Cartesian space columns are present
    space_cols = sorted(_WHERE_CARTESIAN & set(df.columns))

    # Preserve keypoint order as it appears in the file (first occurrence)
    seen: set[str] = set()
    keypoint_names: list[str] = []
    for kp in df["keypoint"].astype(str):
        if kp not in seen:
            keypoint_names.append(kp)
            seen.add(kp)

    individual_names = sorted(df["individual"].astype(str).unique().tolist())
    time_values = sorted(df["time"].unique().tolist())

    n_frames = len(time_values)
    n_space = len(space_cols)
    n_keypoints = len(keypoint_names)
    n_individuals = len(individual_names)

    position_array = np.full(
        (n_frames, n_space, n_keypoints, n_individuals), np.nan
    )
    has_confidence = "confidence" in df.columns
    confidence_array = (
        np.full((n_frames, n_keypoints, n_individuals), np.nan)
        if has_confidence
        else None
    )

    # Build fast integer-index lookups then fill arrays in one vectorised pass
    time_idx = {t: i for i, t in enumerate(time_values)}
    ind_idx = {ind: i for i, ind in enumerate(individual_names)}
    kp_idx = {kp: i for i, kp in enumerate(keypoint_names)}

    ti = df["time"].map(time_idx).to_numpy()
    ii = df["individual"].astype(str).map(ind_idx).to_numpy()
    ki = df["keypoint"].astype(str).map(kp_idx).to_numpy()

    for si, sc in enumerate(space_cols):
        position_array[ti, si, ki, ii] = df[sc].to_numpy()

    if has_confidence:
        confidence_array[ti, ki, ii] = df["confidence"].to_numpy()  # type: ignore[index]

    fps_to_use = _resolve_fps(fps, meta)
    source = meta.get("source") or None

    ds = from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps_to_use,
        source_software=source,
    )
    _attach_extra_attrs(ds, meta, file_path)
    logger.info(f"Loaded poses dataset from {file_path.name}")
    return ds


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #


def _decode_aniframe_metadata(file_path: Path) -> dict[str, Any]:
    """Decode aniframe metadata from a Parquet file's schema metadata.

    Parameters
    ----------
    file_path
        Path to the aniframe Parquet file.

    Returns
    -------
    dict
        Decoded metadata fields. Returns an empty dict if metadata
        cannot be read or decoded.

    """
    schema_meta = pq.read_schema(file_path).metadata
    if schema_meta is None or b"r" not in schema_meta:
        logger.warning(
            f"No aniframe metadata found in {file_path.name}. "
            "Source software, fps, and units will not be set from "
            "file metadata."
        )
        return {}

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Missing constructor for R class",
                category=UserWarning,
            )
            r_obj = rdata.read_rds(io.BytesIO(schema_meta[b"r"]))
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Could not decode aniframe metadata in {file_path.name}: {e}. "
            "Source software, fps, and units will not be set from "
            "file metadata."
        )
        return {}

    return _flatten_r_metadata(r_obj)


def _flatten_r_metadata(r_obj: Any) -> dict[str, Any]:
    """Convert an rdata-parsed R named list to a plain Python dict.

    aniframe serialises its metadata as an ``aniframe_metadata`` R object
    that rdata decodes as a dict with an ``attributes`` key containing
    a nested ``metadata`` dict.  This function navigates that structure
    and falls back to treating the top-level dict as flat metadata when
    the nested path is absent.

    Parameters
    ----------
    r_obj
        The R object returned by :func:`rdata.read_rds`. Expected to be a
        dict-like named R list (possibly nested).

    Returns
    -------
    dict
        Metadata fields mapped to plain Python values.

    """
    if not isinstance(r_obj, dict):
        return {}
    # aniframe metadata is nested: r_obj["attributes"]["metadata"]
    attrs = r_obj.get("attributes")
    if isinstance(attrs, dict):
        meta = attrs.get("metadata")
        if isinstance(meta, dict):
            return {str(k): _r_value_to_python(v) for k, v in meta.items()}
    # Fallback: top-level dict is already flat metadata
    return {str(k): _r_value_to_python(v) for k, v in r_obj.items()}


def _r_value_to_python(value: Any) -> Any:
    """Convert a single rdata-decoded R value to a plain Python type.

    Handles the common R types produced by aniframe metadata:

    - ``pandas.Categorical`` (R factor) → string of the first level value
    - ``numpy.ndarray`` (R vector) → scalar or list of Python values
    - ``float``/``None`` R NA → ``None``
    - Plain Python scalars → returned as-is

    Parameters
    ----------
    value
        A value decoded by :mod:`rdata`.

    Returns
    -------
    Any
        A plain Python scalar, list, or ``None``.

    """
    if value is None:
        return None
    if isinstance(value, pd.Categorical):
        return _r_categorical_to_python(value)
    if isinstance(value, np.ndarray):
        return _r_ndarray_to_python(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _r_categorical_to_python(value: pd.Categorical) -> str | None:
    """Extract the string value from a pandas Categorical (R factor).

    Parameters
    ----------
    value
        A single-element :class:`pandas.Categorical` decoded by rdata.

    Returns
    -------
    str or None

    """
    if len(value) == 0:
        return None
    v = value[0]
    return None if pd.isna(v) else str(v)


def _r_scalar_to_python(v: Any) -> Any:
    """Convert a single rdata element to a plain Python value or ``None``."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v.item() if isinstance(v, np.generic) else v


def _r_ndarray_to_python(value: np.ndarray) -> Any:
    """Convert a numpy array decoded by rdata to a Python scalar or list.

    Parameters
    ----------
    value
        A :class:`numpy.ndarray` decoded by rdata from an R vector.

    Returns
    -------
    Any
        A Python scalar for length-1 arrays, a list for longer arrays,
        or ``None`` for empty arrays or NA values.

    """
    if value.size == 0:
        return None
    if value.ndim == 0 or value.size == 1:
        return _r_scalar_to_python(value.flat[0])
    return [_r_scalar_to_python(v) for v in value]


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and validate aniframe columns for movement compatibility.

    Renames ``track`` to ``individual`` with a warning if needed.
    Drops extra ``variables_what`` / ``variables_when`` columns that have
    only one unique value (logged at INFO level). Raises if any such column
    has multiple unique values.

    Parameters
    ----------
    df
        The raw DataFrame read from the Parquet file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with canonical column names and extra columns removed.

    Raises
    ------
    ValueError
        If an extra identity or temporal column holds more than one unique
        value.

    """
    cols = set(df.columns)

    if "track" in cols and "individual" not in cols:
        warnings.warn(
            "Column 'track' has been renamed to 'individual' for "
            "compatibility with movement.",
            UserWarning,
            stacklevel=4,
        )
        df = df.rename(columns={"track": "individual"})
        cols = set(df.columns)

    # Columns that are recognised but not canonical in movement
    extra_what = (_WHAT_COLS - {"individual", "keypoint"}) & cols
    extra_when = (_WHEN_COLS - {"time"}) & cols

    for col in sorted(extra_what | extra_when):
        n_unique = df[col].nunique()
        if n_unique == 1:
            logger.info(
                f"Column '{col}' has a single unique value "
                f"({df[col].iloc[0]!r}) and will be ignored."
            )
            df = df.drop(columns=[col])
        else:
            raise logger.error(
                ValueError(
                    f"Column '{col}' contains {n_unique} unique values. "
                    "movement cannot represent hierarchical identity or "
                    "multi-level time contexts. "
                    "Only files where this column holds a single value "
                    "can be loaded."
                )
            )

    return df


def _check_no_polar_coords(cols: set[str]) -> None:
    """Raise if polar or spherical coordinate columns are present.

    Parameters
    ----------
    cols
        Set of column names in the DataFrame.

    Raises
    ------
    ValueError
        If any of ``rho``, ``phi``, or ``theta`` are present.

    """
    unsupported = _WHERE_POLAR & cols
    if unsupported:
        raise logger.error(
            ValueError(
                f"Polar/spherical coordinate columns {sorted(unsupported)} "
                "are not supported. movement only supports Cartesian "
                "coordinate systems (x, y, z)."
            )
        )


def _resolve_fps(
    fps_override: float | None,
    meta: dict[str, Any],
) -> float | None:
    """Determine the fps value to pass to ``from_numpy``.

    An explicitly provided ``fps_override`` always takes precedence.
    Otherwise, ``sampling_rate`` from the aniframe metadata is used when
    the ``unit_time`` is not ``"frame"``.

    Parameters
    ----------
    fps_override
        fps value supplied by the caller (``None`` if not provided).
    meta
        Decoded aniframe metadata dict.

    Returns
    -------
    float or None
        fps to pass to :func:`from_numpy`, or ``None`` for frame-number
        time coordinates.

    """
    if fps_override is not None:
        return fps_override

    unit_time = meta.get("unit_time")
    sampling_rate = meta.get("sampling_rate")

    # Frame numbers: no fps needed
    if unit_time is None or unit_time == "frame":
        return None

    # All other recognised time units → derive fps from sampling_rate
    if unit_time in _TIME_UNIT_TO_SECONDS or unit_time == "s":
        if sampling_rate is not None:
            try:
                return float(sampling_rate)
            except (TypeError, ValueError):
                pass
        logger.warning(
            f"unit_time is '{unit_time}' but sampling_rate is not available "
            "in the metadata. Time coordinates will be in frame numbers."
        )
        return None

    logger.warning(
        f"Unrecognised unit_time '{unit_time}'. "
        "Time coordinates will be in frame numbers."
    )
    return None


def _attach_extra_attrs(
    ds: xr.Dataset,
    meta: dict[str, Any],
    file_path: Path,
) -> None:
    """Attach aniframe-specific metadata to the dataset attributes.

    Parameters
    ----------
    ds
        The ``movement`` dataset to annotate (modified in-place).
    meta
        Decoded aniframe metadata dict.
    file_path
        Path to the source file.

    """
    ds.attrs["source_file"] = file_path.as_posix()

    for attr_key, meta_key in (
        ("space_unit", "unit_space"),
        ("reference_frame", "reference_frame"),
        ("point_of_reference", "point_of_reference"),
    ):
        value = meta.get(meta_key)
        if value is not None:
            ds.attrs[attr_key] = value

    por = meta.get("point_of_reference")
    if por == "bottom_left":
        warnings.warn(
            "The aniframe file uses a bottom-left coordinate origin "
            "('point_of_reference' = 'bottom_left'). movement and napari "
            "conventionally use a top-left origin. "
            "Y-axis values may appear flipped.",
            UserWarning,
            stacklevel=4,
        )
