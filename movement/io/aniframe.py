"""Helpers for loading aniframe Parquet files into ``movement``.

The public loader entry point :func:`movement.io.load_poses.from_aniframe_file`
lives alongside the other format-specific loaders in
:mod:`movement.io.load_poses`. This module collects the private helpers and
constants used to decode aniframe metadata and assemble the resulting
``movement`` poses dataset.
"""

import io
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from movement.utils.logging import logger

# pyarrow and rdata are optional dependencies (the 'aniframe' extra). Defer
# their import errors so that ``import movement`` works without them, and
# raise a clear, install-instruction-bearing ImportError only when an
# aniframe-specific code path is actually invoked.
_ANIFRAME_DEPS_INSTALL_HINT = (
    "Reading aniframe Parquet files requires the optional 'aniframe' "
    "dependencies (pyarrow, rdata), which are not installed.\n\n"
    "Install them with one of:\n"
    "  pip install 'movement[aniframe]'\n"
    "  uv pip install 'movement[aniframe]'\n"
    "  conda install -c conda-forge pyarrow rdata\n\n"
    "See https://movement.neuroinformatics.dev/latest/user_guide/"
    "installation.html for details."
)

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore[assignment]

try:
    import rdata
except ImportError:
    rdata = None  # type: ignore[assignment]


def _require_aniframe_extras() -> None:
    """Raise ImportError if pyarrow or rdata is missing."""
    if pq is None or rdata is None:
        raise ImportError(_ANIFRAME_DEPS_INSTALL_HINT)


_WHERE_CARTESIAN: frozenset[str] = frozenset({"x", "y", "z"})
_WHERE_POLAR: frozenset[str] = frozenset({"rho", "phi", "theta"})

# Ordered dimension subsets used when inferring dimensionality of extra
# data columns. Checked from coarsest (constant) to finest (all axes).
_EXTRA_DIM_CANDIDATES: tuple[tuple[str, ...], ...] = (
    (),
    ("time",),
    ("keypoints",),
    ("individuals",),
    ("time", "keypoints"),
    ("time", "individuals"),
    ("keypoints", "individuals"),
    ("time", "keypoints", "individuals"),
)

# Maps movement dimension names to the corresponding DataFrame column names.
_DIM_TO_DF_COL: dict[str, str] = {
    "time": "time",
    "keypoints": "keypoint",
    "individuals": "individual",
}

# Conversion factors from each time unit to seconds
_TIME_UNIT_TO_SECONDS: dict[str, float] = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


def _decode_aniframe_metadata(file_path: Path) -> dict[str, Any]:
    """Decode aniframe metadata from a Parquet file's schema metadata.

    Parameters
    ----------
    file_path
        Path to the aniframe Parquet file.

    Returns
    -------
    dict
        Decoded metadata fields.

    Raises
    ------
    ImportError
        If the optional ``aniframe`` extras (``pyarrow``, ``rdata``) are
        not installed.
    ValueError
        If the Parquet schema lacks the aniframe metadata key (``b"r"``)
        or the embedded R blob cannot be parsed.

    """
    _require_aniframe_extras()
    schema_meta = pq.read_schema(file_path).metadata
    if schema_meta is None or b"r" not in schema_meta:
        raise logger.error(
            ValueError(
                f"No aniframe metadata found in {file_path.name}. "
                "The expected aniframe metadata key 'r' was not found "
                "in the file's Parquet schema metadata."
            )
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Missing constructor for R class",
                category=UserWarning,
            )
            r_obj = rdata.read_rds(io.BytesIO(schema_meta[b"r"]))
    except Exception as e:
        raise logger.error(
            ValueError(
                f"Could not decode aniframe metadata in {file_path.name}: "
                f"{e}. The 'r' metadata key is present but the embedded "
                "R blob could not be parsed as an aniframe metadata object."
            )
        ) from e

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
        The first category's string value, or ``None`` if the value is empty
        or NA.

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


def _resolve_columns(
    df: pd.DataFrame,
    vars_what: list[str],
    vars_when: list[str],
    vars_where: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Rename and validate aniframe columns for movement compatibility.

    Renames ``track`` to ``individual`` (logged at INFO level) if needed.
    Drops extra identity/temporal columns that hold a single unique value
    (logged at INFO level). Raises if any such column has multiple unique
    values. Returns the cleaned DataFrame alongside a sorted list of extra
    data columns — those not belonging to any aniframe variable category —
    which callers may add as supplementary Dataset variables.

    Parameters
    ----------
    df
        The raw DataFrame read from the Parquet file.
    vars_what
        Column names classified as identity variables by the aniframe
        metadata (``variables_what``).
    vars_when
        Column names classified as temporal variables by the aniframe
        metadata (``variables_when``).
    vars_where
        Column names classified as coordinate variables by the aniframe
        metadata (``variables_where``).

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        ``(df, extra_data_cols)`` where *df* has canonical column names and
        extra identity/temporal columns removed, and *extra_data_cols* is a
        sorted list of column names that do not belong to any aniframe
        variable category (e.g. derived measurements such as ``speed``).

    Raises
    ------
    ValueError
        If an extra identity or temporal column holds more than one unique
        value.

    """
    cols = set(df.columns)

    if "track" in cols and "individual" not in cols:
        logger.info(
            "Column 'track' has been interpreted as 'individual'. "
            "movement treats tracks as individuals. If your data contains "
            "multiple tracks per individual, stitch them before loading."
        )
        df = df.rename(columns={"track": "individual"})
        cols = set(df.columns)

    # All columns that belong to a known aniframe or movement category.
    # Hardcoding the movement canonical names and all coordinate names
    # ensures they are never mistaken for extra data columns, even when the
    # metadata vars lists use aliases (e.g. "track" instead of "individual").
    known_cols: set[str] = (
        set(vars_what)
        | set(vars_when)
        | set(vars_where)
        | _WHERE_CARTESIAN
        | _WHERE_POLAR
        | {"individual", "keypoint", "track", "confidence"}
    )
    extra_data_cols = sorted(cols - known_cols)

    # Extra identity/temporal columns (classified by metadata but not
    # canonical in movement): drop if constant, error if multi-valued.
    canonical_what = {"individual", "keypoint"}
    canonical_when = {"time"}
    extra_what = (set(vars_what) - canonical_what - {"track"}) & cols
    extra_when = (set(vars_when) - canonical_when) & cols

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

    return df, extra_data_cols


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

    # All recognised time units → derive fps from sampling_rate
    if unit_time in _TIME_UNIT_TO_SECONDS:
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


def _flip_y_to_top_left(
    position_array: np.ndarray,
    meta: dict[str, Any],
    space_cols: list[str],
) -> float | None:
    """Flip y values in-place when the file uses a bottom-left origin.

    movement and napari use a top-left image-coordinate origin. aniframe
    files often store y in a bottom-left convention; this helper converts
    them via ``new_y = y_height - y`` so the data overlays correctly on
    the source video/frame.

    The conversion height is taken from the ``y_height`` metadata field
    when present (this is what ``animovement`` writes). If absent — e.g.
    older files or third-party producers — it falls back to
    ``max(y)`` from the data and logs an INFO message so the choice is
    visible.

    Parameters
    ----------
    position_array
        Array of shape ``(n_frames, n_space, n_keypoints, n_individuals)``,
        modified in-place along the y axis.
    meta
        Decoded aniframe metadata dict.
    space_cols
        Cartesian space columns present in the file (e.g. ``["x", "y"]``
        or ``["x", "y", "z"]``), in the same order as the position
        array's space axis.

    Returns
    -------
    float or None
        The ``y_height`` value used for the flip, or ``None`` when the
        flip was skipped (no y axis in the data).

    """
    if "y" not in space_cols:
        logger.info(
            "Origin is 'bottom_left' but the file has no y column; "
            "skipping the bottom_left → top_left flip."
        )
        return None

    y_index = space_cols.index("y")
    y_height_meta = meta.get("y_height")
    if y_height_meta is None:
        y_height = float(np.nanmax(position_array[:, y_index, :, :]))
        logger.info(
            "No 'y_height' in aniframe metadata; using max(y) "
            f"= {y_height} as the height for the bottom_left → "
            "top_left flip."
        )
    else:
        y_height = float(y_height_meta)

    position_array[:, y_index, :, :] = (
        y_height - position_array[:, y_index, :, :]
    )
    logger.info(
        f"Flipped y-axis (bottom_left → top_left) using y_height={y_height}."
    )
    return y_height


def _override_time_coord_from_file(
    ds: xr.Dataset,
    time_values: list[Any],
    meta: dict[str, Any],
) -> xr.Dataset:
    """Replace the ``time`` coord with values read from the file.

    Used when the caller passes ``use_frame_numbers_from_file=True`` to
    preserve original time stamps (including irregular sampling). The
    ``unit_time`` metadata field decides how raw values are mapped to the
    dataset's ``time`` coord:

    - ``"frame"`` → kept as ``int64`` frame numbers; ``time_unit="frames"``.
    - any unit in :data:`_TIME_UNIT_TO_SECONDS` → multiplied by the
      corresponding factor to land in seconds; ``time_unit="seconds"``.
    - missing or unrecognised → values kept as ``float64`` and
      ``time_unit="seconds"`` is assumed; an INFO log is emitted.

    Parameters
    ----------
    ds
        Dataset returned by :func:`from_numpy`.
    time_values
        Sorted list of unique values from the file's ``time`` column.
    meta
        Decoded aniframe metadata dict.

    Returns
    -------
    xarray.Dataset
        Dataset with the ``time`` coord and ``time_unit`` attribute updated.

    """
    unit_time = meta.get("unit_time")
    time_arr = np.asarray(time_values)

    if unit_time == "frame":
        time_arr = time_arr.astype(np.int64)
        time_unit = "frames"
    elif unit_time in _TIME_UNIT_TO_SECONDS:
        time_arr = (
            time_arr.astype(np.float64) * _TIME_UNIT_TO_SECONDS[unit_time]
        )
        time_unit = "seconds"
    else:
        if unit_time is not None:
            logger.info(
                f"Unrecognised unit_time '{unit_time}'. Time values from "
                "the file will be kept without conversion."
            )
        time_arr = time_arr.astype(np.float64)
        time_unit = "seconds"

    ds = ds.assign_coords(time=time_arr)
    ds.attrs["time_unit"] = time_unit
    return ds


def _attach_extra_attrs(
    ds: xr.Dataset,
    meta: dict[str, Any],
    file_path: Path,
    flipped_y_height: float | None = None,
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
    flipped_y_height
        Height value used by :func:`_flip_y_to_top_left`, or ``None`` if
        no flip was applied. When non-``None``, the dataset's ``origin``
        attribute is set to ``"top_left"`` and ``y_height`` is recorded
        so the flip can be reversed later.

    """
    ds.attrs["source_file"] = file_path.as_posix()

    for attr_key, meta_key in (
        ("space_unit", "unit_space"),
        ("reference_frame", "reference_frame"),
    ):
        value = meta.get(meta_key)
        if value is not None:
            ds.attrs[attr_key] = value

    if flipped_y_height is not None:
        # Data has been converted to top_left; advertise the new origin
        # and preserve y_height so callers can roundtrip back if needed.
        ds.attrs["origin"] = "top_left"
        ds.attrs["y_height"] = flipped_y_height
    else:
        origin = meta.get("origin")
        if origin is not None:
            ds.attrs["origin"] = origin
        y_height_meta = meta.get("y_height")
        if y_height_meta is not None:
            ds.attrs["y_height"] = float(y_height_meta)


def _extract_meta_vars(
    meta: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Extract the variable-classification lists from metadata.

    aniframe metadata carries ``variables_what``, ``variables_when``, and
    ``variables_where`` fields that classify every DataFrame column into an
    identity, temporal, or coordinate role. Their presence is enforced by
    :class:`movement.validators.files.ValidAniframeParquet`; this helper
    only normalises each to a plain ``list[str]`` (R length-1 vectors are
    decoded as scalars rather than lists by :func:`_r_ndarray_to_python`).

    Parameters
    ----------
    meta
        Decoded aniframe metadata dict.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``(vars_what, vars_when, vars_where)`` as lists of strings.

    """

    def _to_list(val: Any) -> list[str]:
        if isinstance(val, list):
            return [str(v) for v in val]
        return [str(val)]

    return (
        _to_list(meta["variables_what"]),
        _to_list(meta["variables_when"]),
        _to_list(meta["variables_where"]),
    )


def _infer_extra_dims(df: pd.DataFrame, col: str) -> tuple[str, ...]:
    """Return the minimum xarray dimensions needed to represent a column.

    Checks ordered subsets of ``{"time", "keypoints", "individuals"}`` from
    coarsest (constant) to finest (all three axes), and returns the first
    subset whose groupby uniquely determines every value of ``col``.

    Parameters
    ----------
    df
        Long-format DataFrame containing ``time``, ``keypoint``, and
        ``individual`` columns.
    col
        Name of the extra data column to inspect.

    Returns
    -------
    tuple[str, ...]
        Movement dimension names (e.g. ``("time", "individuals")``).
        Empty tuple if the column is constant across every row.

    """
    for dims in _EXTRA_DIM_CANDIDATES:
        if not dims:
            if df[col].nunique() <= 1:
                return ()
            continue
        group_cols = [_DIM_TO_DF_COL[d] for d in dims]
        if df.groupby(group_cols)[col].nunique().max() <= 1:
            return dims
    return ("time", "keypoints", "individuals")


def _build_extra_array(
    df: pd.DataFrame,
    col: str,
    dims: tuple[str, ...],
    *,
    ti: np.ndarray,
    ki: np.ndarray,
    ii: np.ndarray,
    n_frames: int,
    n_keypoints: int,
    n_individuals: int,
) -> np.ndarray:
    """Build a numpy array for an extra data column given its dimensions.

    Numeric columns (excluding bool) are cast to ``float64`` with ``nan``
    fill. Boolean and other dtypes (strings, mixed objects) use an
    ``object`` array with ``None`` fill so True/False values are preserved
    rather than coerced to 0.0/1.0. When multiple rows map to the same
    index (because the column does not vary along the omitted axes), the
    last written value is kept; this is always correct when the column's
    dimensionality was correctly inferred.

    Parameters
    ----------
    df
        Long-format DataFrame.
    col
        Name of the column to extract.
    dims
        Movement dimension names as returned by :func:`_infer_extra_dims`.
    ti, ki, ii
        Integer row-index arrays for the ``time``, ``keypoints``, and
        ``individuals`` dimensions respectively.
    n_frames, n_keypoints, n_individuals
        Axis sizes.

    Returns
    -------
    numpy.ndarray
        Array with shape determined by ``dims``.

    """
    _dim_idx: dict[str, np.ndarray] = {
        "time": ti,
        "keypoints": ki,
        "individuals": ii,
    }
    _dim_size: dict[str, int] = {
        "time": n_frames,
        "keypoints": n_keypoints,
        "individuals": n_individuals,
    }
    series = df[col]
    if pd.api.types.is_bool_dtype(series):
        # is_numeric_dtype also returns True for bool, so check this first
        # to avoid silently coercing True/False to 0.0/1.0 with NaN fill.
        fill: Any = None
        dtype: Any = object
        vals = series.to_numpy(dtype=object)
    elif pd.api.types.is_numeric_dtype(series):
        fill = np.nan
        dtype = np.float64
        vals = series.to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        fill = None
        dtype = object
        vals = series.to_numpy(dtype=object)
    shape = tuple(_dim_size[d] for d in dims)
    arr = np.full(shape, fill, dtype=dtype)
    idx = tuple(_dim_idx[d] for d in dims)
    arr[idx] = vals
    return arr
