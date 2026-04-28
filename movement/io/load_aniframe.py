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

_VALID_EXTRA_VAR_DIMS: frozenset[str] = frozenset(
    {"time", "keypoints", "individuals"}
)

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
    file: str | Path,
    fps: float | None = None,
    extra_var_dims: str | tuple[str, ...] = (),
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
    extra_var_dims
        Minimum set of dimensions that every extra data variable must have.
        Dimensions are automatically inferred from the data; any dimension
        listed here that is not already in the inferred set will be added.
        Valid values are ``"time"``, ``"keypoints"``, and ``"individuals"``.
        A single dimension may be passed as a plain string (e.g.
        ``"individuals"``) or as a one-element tuple (``("individuals",)``).
        Defaults to ``()`` (no floor — pure auto-inference).
        Pass ``"individuals"`` to ensure all extra variables carry the
        ``individuals`` dimension even in single-individual files, where
        auto-inference would otherwise collapse it away.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Raises
    ------
    ValueError
        If the file contains polar or spherical coordinate columns
        (``rho``, ``phi``, ``theta``), if any extra identity or temporal
        column holds more than one unique value, if the aniframe metadata
        is missing the required ``variables_what``, ``variables_when``, or
        ``variables_where`` fields, or if ``extra_var_dims`` contains
        invalid dimension names.

    Warns
    -----
    UserWarning
        If ``point_of_reference`` is ``"bottom_left"`` (which differs from
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

    If the file uses a ``track`` column instead of ``individual``, each
    track is interpreted as an individual and an INFO message is logged.
    If your data contains multiple tracks per individual, stitch them
    before loading.

    Columns that do not belong to any aniframe variable category
    (``variables_what``, ``variables_when``, ``variables_where``) are
    treated as extra data variables and added to the returned Dataset with
    automatically inferred dimensions. Numeric columns are stored as
    ``float64``; all other types are stored as ``object``. Constant columns
    (identical across every row) are skipped.

    The original tracking software is taken from the ``source`` field in
    the aniframe metadata (e.g., ``"SLEAP"`` or ``"DeepLabCut"``).

    """
    extra_var_dims = _normalise_extra_var_dims(extra_var_dims)
    valid_file = cast("ValidFile", file)
    file_path = valid_file.file

    df = pd.read_parquet(file_path)
    meta = _decode_aniframe_metadata(file_path)
    vars_what, vars_when, vars_where = _extract_meta_vars(meta, file_path)
    df, extra_data_cols = _resolve_columns(
        df, vars_what, vars_when, vars_where
    )
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

    # Infer dimensions and build arrays for extra data columns
    extra_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for col in extra_data_cols:
        dims = _infer_extra_dims(df, col)
        if not dims:
            logger.info(f"Column '{col}' is constant and will be skipped.")
            continue
        if extra_var_dims:
            dims = tuple(
                d
                for d in ("time", "keypoints", "individuals")
                if d in dims or d in extra_var_dims
            )
        extra_vars[col] = (
            dims,
            _build_extra_array(
                df,
                col,
                dims,
                ti=ti,
                ki=ki,
                ii=ii,
                n_frames=n_frames,
                n_keypoints=n_keypoints,
                n_individuals=n_individuals,
            ),
        )

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

    for col, (dims, arr) in extra_vars.items():
        ds[col] = xr.DataArray(arr, dims=dims)

    _attach_extra_attrs(ds, meta, file_path)
    logger.info(f"Loaded poses dataset from {file_path.name}")
    return ds


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #


def _normalise_extra_var_dims(
    extra_var_dims: str | tuple[str, ...],
) -> tuple[str, ...]:
    """Normalise and validate the ``extra_var_dims`` parameter.

    Parameters
    ----------
    extra_var_dims
        A single dimension name (string) or a tuple of dimension names.

    Returns
    -------
    tuple[str, ...]
        Normalised tuple of dimension names.

    Raises
    ------
    ValueError
        If any name is not one of ``"time"``, ``"keypoints"``,
        ``"individuals"``.

    """
    if isinstance(extra_var_dims, str):
        extra_var_dims = (extra_var_dims,)
    if invalid := set(extra_var_dims) - _VALID_EXTRA_VAR_DIMS:
        raise logger.error(
            ValueError(
                f"Invalid dimension(s) in extra_var_dims: {sorted(invalid)}."
                f" Valid values are: {sorted(_VALID_EXTRA_VAR_DIMS)}."
            )
        )
    return extra_var_dims


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


def _resolve_columns(
    df: pd.DataFrame,
    vars_what: list[str],
    vars_when: list[str],
    vars_where: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Rename and validate aniframe columns for movement compatibility.

    Renames ``track`` to ``individual`` with a warning if needed.
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


def _extract_meta_vars(
    meta: dict[str, Any],
    file_path: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Extract and validate the variable-classification lists from metadata.

    aniframe metadata carries ``variables_what``, ``variables_when``, and
    ``variables_where`` fields that classify every DataFrame column into an
    identity, temporal, or coordinate role. This function validates that all
    three are present, then normalises each to a plain ``list[str]``
    (R length-1 vectors are decoded as scalars rather than lists by
    :func:`_r_ndarray_to_python`).

    Parameters
    ----------
    meta
        Decoded aniframe metadata dict.
    file_path
        Path to the source file (used in error messages only).

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        ``(vars_what, vars_when, vars_where)`` as lists of strings.

    Raises
    ------
    ValueError
        If any of ``variables_what``, ``variables_when``, or
        ``variables_where`` is absent from the metadata.

    """
    required = ("variables_what", "variables_when", "variables_where")
    missing = [k for k in required if meta.get(k) is None]
    if missing:
        raise logger.error(
            ValueError(
                f"aniframe metadata in {file_path.name} is missing required "
                f"field(s): {missing}. "
                "These fields are needed to classify DataFrame columns."
            )
        )

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

    Numeric columns are cast to ``float64`` with ``nan`` fill. All other
    dtypes (strings, mixed objects) use an ``object`` array with ``None``
    fill. When multiple rows map to the same index (because the column does
    not vary along the omitted axes), the last written value is kept; this
    is always correct when the column's dimensionality was correctly inferred.

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
    if pd.api.types.is_numeric_dtype(series):
        fill: Any = np.nan
        dtype: Any = np.float64
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
