"""Utilities for manipulating dimensions of ``xarray.DataArray`` objects."""

from collections.abc import Iterable

import xarray as xr


def collapse_extra_dimensions(
    da: xr.DataArray,
    preserve_dims: Iterable[int | str] = ("time", "space"),
    **selection: int | str,
) -> xr.DataArray:
    """Collapse a ``DataArray``, preserving only the specified dimensions.

    By default, dimensions that are collapsed retain the corresponding 'slice'
    along their 0th index of those dimensions, unless a particular index for is
    given in the ``selection``.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray of multiple dimensions, which is to be collapsed.
    preserve_dims : Iterable[int | str]
        The dimensions of ``da`` that should not be collapsed.
    selection : int | str
        Mapping from dimension names to a particular index in that dimension.
        Dimensions that appear with an index in ``selection`` retain that index
        slice when collapsed, rather than the default 0th index slice.

    Returns
    -------
    xarray.DataArray
        DataArray whose shape is the same as the shape of the preserved
        dimensions of ``da``, containing the data obtained from a slice along
        the collapsed dimensions.

    Examples
    --------
    Collapse a ``DataArray`` down to just a ``"time"``-``"space"`` slice.

    >>> import xarray as xr
    >>> import numpy as np
    >>> shape = (7, 2, 3, 4)
    >>> da = xr.DataArray(
    ...     data=np.arange(np.prod(shape)).reshape(shape),
    ...     dims=["time", "space", "dim_to_collapse_0", "dim_to_collapse_1"],
    ... )
    >>> space_time = collapse_extra_dimensions(da)
    >>> print(space_time.shape)
    (7, 2)

    The call to ``collapse_extra_dimensions`` above is equivalent to
    ``da.sel(dim_to_collapse_0=0, dim_to_collapse_1=0)``. We can change which
    slice we take from the collapsed dimensions by passing them as keyword
    arguments.

    >>> # Equivalent to da.sel(dim_to_collapse_0=2, dim_to_collapse_1=1)
    >>> space_time_different_slice = collapse_extra_dimensions(
    ...     da, dim_to_collapse_0=2, dim_to_collapse_1=1
    ... )
    >>> print(space_time_different_slice.shape)
    (7, 2)

    We can also change which dimensions are to be preserved.

    >>> time_only = collapse_extra_dims(da, preserve_dims=["time"])
    >>> print(time_only.shape)
    (7,)

    """
    data = da.copy(deep=True)
    dims_to_collapse = [d for d in data.dims if d not in preserve_dims]

    for d in dims_to_collapse:
        index_to_keep = selection.pop(d, 0)
        if isinstance(index_to_keep, int):
            index_to_keep = getattr(data, d)[index_to_keep]
        data = data.sel({d: index_to_keep})
    return data
