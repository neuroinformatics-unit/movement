"""Utilities for manipulating dimensions of ``xarray.DataArray`` objects."""

from collections.abc import Hashable, Iterable

import xarray as xr


def collapse_extra_dimensions(
    da: xr.DataArray,
    preserve_dims: Iterable[str] = ("time", "space"),
    **selection: str,
) -> xr.DataArray:
    """Collapse a ``DataArray``, preserving only the specified dimensions.

    By default, dimensions that are collapsed retain the corresponding 'slice'
    along their 0th index of those dimensions, unless a particular index for is
    given in the ``selection``.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray of multiple dimensions, which is to be collapsed.
    preserve_dims : Iterable[str]
        The dimensions of ``da`` that should not be collapsed.
    selection : str
        Mapping from dimension names to a particular index name in that
        dimension.

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
    >>> shape = (7, 2, 3, 2)
    >>> da = xr.DataArray(
    ...     data=np.arange(np.prod(shape)).reshape(shape),
    ...     dims=["time", "space", "keypoints", "individuals"],
    ...     coords={
    ...         "time": np.arange(7),
    ...         "space": np.arange(2),
    ...         "keypoints": ["nose", "left_ear", "right_ear"],
    ...         "individuals": ["Alice", "Bob"],
    ... )
    >>> space_time = collapse_extra_dimensions(da)
    >>> print(space_time.shape)
    (7, 2)

    The call to ``collapse_extra_dimensions`` above is equivalent to
    ``da.sel(keypoints="head", individuals="Alice")`` (indexing by label).
    We can change which slice we take from the collapsed dimensions by passing
    them as keyword arguments.

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
    make_selection = {
        d: _coord_of_dimension(da, d, selection.pop(d, 0))
        for d in dims_to_collapse
    }
    return data.sel(make_selection)


def _coord_of_dimension(
    da: xr.DataArray, dimension: str, coord_index: int | str
) -> Hashable:
    """Retrieve a coordinate of a given dimension.

    This method handles the case where the coordinate is known by name
    within the coordinates of the ``dimension``.

    If ``coord_index`` is an element of ``da.dimension``, it can just be
    returned. Otherwise, we need to return ``da.dimension[coord_index]``.

    Out of bounds index errors, or non existent dimension errors are handled by
    the underlying ``xarray.DataArray`` implementation.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to retrieve a coordinate from.
    dimension : str
        Dimension of the DataArray to fetch the coordinate from.
    coord_index : int | str
        The index of the coordinate along ``dimension`` to fetch.

    Returns
    -------
    Hashable
        The requested coordinate name at ``da.dimension[coord_index]``.

    """
    dim = getattr(da, dimension)
    return dim[coord_index] if coord_index not in dim else coord_index
