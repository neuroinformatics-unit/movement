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
        Spatial data to check for inclusion within the ``regions``. Must be
        compatible with the ``data`` argument to :func:`contains_point\
        <movement.roi.base.BaseRegionOfInterest.contains_point`.
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

    Notes
    -----
    Regions that have the same name will have a suffix of the form "_XX"
    appended to their names when generating the coordinates. Regions with
    unique names will retain the same name as their corresponding coordinate.

    """
    # Filter out duplicate names if they are provided
    number_of_times_name_appears: defaultdict[str, int] = defaultdict(int)
    for r in regions:
        number_of_times_name_appears[r.name] += 1

    duplicate_names_max_chars = {
        key: int(np.ceil(np.log10(value)).item())
        for key, value in number_of_times_name_appears.items()
        if value > 1
    }
    duplicate_names_used: defaultdict[str, int] = defaultdict(int)

    occupancies = {}
    for r in regions:
        name = r.name
        if name in duplicate_names_max_chars:
            name_suffix = str(duplicate_names_used[name]).zfill(
                duplicate_names_max_chars[name]
            )
            duplicate_names_used[name] += 1
            name = f"{name}_{name_suffix}"
        occupancies[name] = r.contains_point(data)

    return xr.concat(occupancies.values(), dim="occupancy").assign_coords(
        occupancy=list(occupancies.keys())
    )
