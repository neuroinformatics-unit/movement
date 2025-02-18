"""Functions for computing condition arrays involving RoIs."""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import xarray as xr

from movement.roi.base import BaseRegionOfInterest


def compute_region_occupancy(
    data,
    regions: Sequence[BaseRegionOfInterest],
) -> xr.DataArray:
    """Return a condition array indicating if points were inside regions.

    The returned condition array has one extra dimension on top of those in
    ``data``, called ``"occupancy"``. This extra dimension has a number
    of elements equal to the ``regions`` argument, and has coordinates
    corresponding to the names of the given RoIs. For each ``region`` in
    ``regions``, values along this dimension are the result of
    ``region.contains_point(data)``.

    Parameters
    ----------
    data : xarray.DataArray
        Spatial data to check for inclusion within the ``regions``.
    regions : Sequence[BaseRegionOfInterest]
        Regions of Interest that the points in ``data`` will be checked
        against, to see if they lie inside.

    Returns
    -------
    xarray.DataArray
        Output that matches the dimensions of ``data``, except for the
        ``"space"`` dimension which is dropped, and the addition of the
        ``"occupancy"`` dimension that has the same length as the number
        of regions provided. Coordinates along the ``"occupancy"`` dimension
        match the names of the ``regions``. Values along this dimension match
        the output of ``BaseRegionOfInterest.contains_point(data)`` when called
        on the corresponding regions.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from movement.roi import PolygonOfInterest, compute_region_occupancy
    >>> square = PolygonOfInterest(
    ...     [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], name="square"
    ... )
    >>> triangle = PolygonOfInterest(
    ...     [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], name="triangle"
    ... )
    >>> data = xr.DataArray(
    ...     data=np.array([[0.25, 0.25], [0.75, 0.75]]),
    ...     dims=["time", "space"],
    ...     coords={"space": ["x", "y"]},
    ... )
    >>> occupancies = compute_region_occupancy(data, [square, triangle])
    >>> occupancies.sel(occupancy="square").values
    np.array([True, True])
    >>> occupancies.sel(occupancy="triangle").values
    np.array([True, False])

    """
    # Filter out duplicate names if they are provided
    duplicate_names_count: defaultdict[str, int] = defaultdict(int)
    for r in regions:
        duplicate_names_count[r.name] += 1
    duplicate_names_max_chars = {
        key: np.ceil(np.log10(value)) + 1
        for key, value in duplicate_names_count.items()
    }
    duplicate_names_used: defaultdict[str, int] = defaultdict(int)

    occupancies = {}
    for r in regions:
        name = r.name
        if name in duplicate_names_max_chars:
            name_suffix = str(duplicate_names_used[name]).zfill(
                duplicate_names_max_chars[name]
            )
            name = f"{name}_{name_suffix}"
            duplicate_names_used[name] += 1
        occupancies[name] = r.contains_point(data)

    return xr.concat(
        occupancies.values(), dim="region occupancy", coords=occupancies.keys()
    )
