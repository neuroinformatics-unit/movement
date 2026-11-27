"""Convert ``movement`` data to ``pynapple`` objects.

``pynapple`` is an optional dependency. Install it with
``pip install "movement[pynapple]"``.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from movement.utils.logging import logger
from movement.validators.arrays import validate_dims_coords
from movement.validators.datasets import ValidPosesInputs

if TYPE_CHECKING:
    from collections.abc import Hashable

    import pynapple as nap
    import xarray as xr

# Canonical non-time dims, in reverse order
# Their labels end up in column metadata.
_METADATA_DIMS = tuple(
    d for d in reversed(ValidPosesInputs.DIM_NAMES) if d != "time"
)
# Dims stacked into columns: as above, but without ``individual``
_COLUMN_DIMS = tuple(d for d in _METADATA_DIMS if d != "individual")


def _import_pynapple():
    """Import pynapple, raising an informative error if it is missing."""
    try:
        import pynapple as nap
    except ImportError as e:
        raise logger.error(
            ImportError(
                "This conversion requires the pynapple package. "
                'Install it with: pip install "movement[pynapple]"'
            )
        ) from e
    return nap


def _validate_dims(da: xr.DataArray) -> xr.DataArray:
    """Check that ``da`` holds one individual's data with canonical dims."""
    validate_dims_coords(da, {"time": []})  # time is required

    n_individuals = da.sizes.get("individual", 1)
    # raise an informative error for multi-individual
    if n_individuals > 1:
        raise logger.error(
            ValueError(
                "Only one individual can be converted at a time, but the "
                f"DataArray contains {n_individuals} individuals. Select "
                'one first, e.g. da.sel(individual="mouse1").'
            )
        )
    # drop the size-1 individual dimension, keeping its label as a
    # scalar coordinate
    if "individual" in da.dims:
        da = da.squeeze("individual")

    # Extra (non-canonical) dims are not supported
    unsupported = [d for d in da.dims if d not in ("time", *_COLUMN_DIMS)]
    if unsupported:
        raise logger.error(
            ValueError(
                f"Unsupported dimensions {unsupported}. Only "
                f"{list(ValidPosesInputs.DIM_NAMES)} are supported. "
                "Select a single label along the other dimensions first."
            )
        )
    return da


def to_pynapple_tsdframe(da: xr.DataArray) -> nap.TsdFrame:
    """Convert a ``movement`` DataArray to a ``pynapple`` TsdFrame.

    Only one individual can be converted at a time, so select one first
    if the data contain several (see Examples). The DataArray must have a
    ``time`` dimension, in seconds (see Notes).

    Parameters
    ----------
    da : xarray.DataArray
        A DataArray with a ``time`` dimension. It may only have
        ``movement``'s canonical dimensions (``time``, ``space``,
        ``keypoint``, ``individual``) and at most one individual.

    Returns
    -------
    pynapple.TsdFrame
        A TsdFrame of shape ``(n_time, n_columns)``. Column names are
        derived from the ``keypoint`` and ``space`` labels, or from the
        DataArray's name if neither dimension is present (see Notes).

    Raises
    ------
    ValueError
        If ``da`` has no ``time`` dimension, contains more than one
        individual, or has non-canonical dimensions (e.g. ``view``).
    ImportError
        If ``pynapple`` is not installed.

    Notes
    -----
    **Time units.**
    ``pynapple`` interprets timestamps as seconds, so the ``time``
    coordinate is assumed to be in seconds. If it is in frames, convert it
    first, using the frame rate ``fps``:

    >>> da = da.assign_coords(time=da.time / fps)

    **Column names and metadata.**
    If the ``keypoint`` and/or ``space`` dimensions are present, column
    names are derived from their labels (e.g. ``snout_x``, ``snout`` or
    ``x``; see the first example). If ``time`` is the only dimension, the
    DataArray's name is used instead (e.g. ``speed``), falling back to
    ``Unnamed`` if it has no name. To choose the name yourself, rename the
    DataArray before converting it:

    >>> tsdframe = to_pynapple_tsdframe(da.rename("my_variable"))

    Each column also carries ``pynapple`` metadata: the labels it was
    derived from, stored as one field per dimension. Canonical dimensions
    with a single label, such as ``individual`` after
    ``.sel(individual="mouse1")``, are included too, with the same value
    for every column. For example, converting
    ``ds.position.sel(individual="mouse1")`` gives this metadata:

    =========  ========  =====  ==========
    column     keypoint  space  individual
    =========  ========  =====  ==========
    snout_x    snout     x      mouse1
    snout_y    snout     y      mouse1
    ...        ...       ...    ...
    =========  ========  =====  ==========

    Printing a TsdFrame shows only the first two metadata fields, but the
    others (here ``individual``) are still there. Access the whole table
    via ``tsdframe.metadata``, or a single field as an attribute
    (e.g. ``tsdframe.keypoint``). Columns can be selected by metadata as
    well as by name (see Examples).

    **Missing values.** NaNs are kept as they are, even though they are
    not supported by all ``pynapple`` functions. Most notably,
    ``pynapple.compute_tuning_curves`` raises an error if the features
    contain NaNs and no ``range`` is given. To avoid this, either fill
    gaps with :func:`movement.filtering.interpolate_over_time` before
    converting, or use ``dropna()`` on the desired TsdFrame columns
    afterwards:

    >>> nose = tsdframe[["snout_x", "snout_y"]].dropna()

    Examples
    --------
    Load a sample dataset of two mice, labelled ``"1"`` and ``"2"``, in an
    octagon arena (time is in seconds):

    >>> from movement import sample_data
    >>> from movement.io import to_pynapple_tsdframe
    >>> ds = sample_data.fetch_dataset("SLEAP_two-mice_octagon.analysis.h5")

    Convert the position of mouse ``"1"``:

    >>> pos = to_pynapple_tsdframe(ds.position.sel(individual="1"))
    >>> pos
    Time (s)    Nose_x             Nose_y             ...
    ----------  -----------------  -----------------  -----
    0.0         796.47314453125    843.9411010742188  ...
    0.02        784.9976196289062  840.1928100585938  ...
    ...
    Metadata
    keypoint    Nose               Nose               ...
    space       x                  y                  ...
    ...         ...                ...                ...
    dtype: float32, shape: (9000, 14)

    Select its nose by column name or via column metadata:

    >>> pos[["Nose_x", "Nose_y"]]  # select by column name
    >>> pos[:, pos.keypoint == "Nose"]  # select by metadata

    Convert a derived variable, such as speed:

    >>> from movement.kinematics import compute_speed
    >>> speed = to_pynapple_tsdframe(
    ...     compute_speed(ds.position.sel(individual="1"))
    ... )

    Convert each individual separately, storing the results in a dictionary
    keyed by individual:

    >>> positions = {
    ...     ind: to_pynapple_tsdframe(ds.position.sel(individual=ind))
    ...     for ind in ds.individual.values
    ... }
    >>> positions["2"]  # access the TsdFrame for mouse "2"

    """
    nap = _import_pynapple()
    da = _validate_dims(da)
    stacked = [d for d in _COLUMN_DIMS if d in da.dims]
    da = da.transpose("time", *stacked)
    data = da.values.reshape(da.sizes["time"], -1)
    labels = list(
        itertools.product(
            *([str(label) for label in da[d].values] for d in stacked)
        )
    )
    # One metadata field per stacked dimension, then one constant field
    # per canonical dimension with a single (scalar) label
    metadata = {
        dim: [column_labels[i] for column_labels in labels]
        for i, dim in enumerate(stacked)
    }
    metadata |= {
        dim: [str(da[dim].item())] * len(labels)
        for dim in _METADATA_DIMS
        if dim in da.coords and dim not in da.dims
    }
    columns: list[Hashable]
    if stacked:
        columns = ["_".join(column_labels) for column_labels in labels]
    else:
        columns = [da.name if da.name is not None else "Unnamed"]
    return nap.TsdFrame(
        t=da.time.values, d=data, columns=columns, metadata=metadata or None
    )
