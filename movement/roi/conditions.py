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

    The function returns a boolean DataArray where each element indicates
    whether a point in the input ``data`` lies within the corresponding RoIs
    in ``regions``. The original dimensions of ``data`` are preserved, except
    for the ``space`` dimension which is replaced by the ``region``
    dimension. The ``region`` dimension has a number of elements equal to
    the number of RoIs in the ``regions`` argument and it's coordinate names
    correspond to the names of the given RoIs.

    Parameters
    ----------
    data : xarray.DataArray
        Spatial data to check for inclusion within the ``regions``. Must be
        compatible with the ``data`` argument to :func:`contains_point\
        <movement.roi.base.BaseRegionOfInterest.contains_point>`.
    regions : Sequence[BaseRegionOfInterest]
        Regions of Interest that the points in ``data`` will be checked
        against, to see if they lie inside.

    Returns
    -------
    xarray.DataArray
        A boolean ``DataArray`` providing occupancy information.

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
    >>> occupancies.sel(region="square").values
    np.array([True, True])
    >>> occupancies.sel(region="triangle").values
    np.array([True, False])

    Notes
    -----
    When RoIs in ``regions`` have identical names, a suffix
    will be appended to their name in the form of "_X", where "X" is a number
    starting from 0. These numbers are zero-padded depending on the maximum
    number of regions with identical names (e.g. if there are 100 RoIs with the
    same name, "00" will be appended to the first of them)

    Regions with unique names will retain their original name as their
    corresponding coordinate name.

    """
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

    return xr.concat(occupancies.values(), dim="region").assign_coords(
        region=list(occupancies.keys())
    )
