"""Functions for computing condition arrays involving RoIs."""

from collections import defaultdict
from typing import Literal

import numpy as np
import xarray as xr

from movement.roi.io import ROICollection


def compute_region_occupancy(
    data,
    regions: ROICollection,
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
    data
        Spatial data to check for inclusion within the ``regions``. Must be
        compatible with the ``position`` argument to :func:`contains_point()\
        <movement.roi.BaseRegionOfInterest.contains_point>`.
    regions
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


def compute_entry_exits(
    data: xr.DataArray,
    regions: ROICollection,
    mode: Literal["centroid", "all", "any", "majority"] = "centroid",
    min_frames: int = 1,
) -> xr.DataArray:
    """Return an array indicating when keypoints entered or exited regions.

    Detects transitions into and out of Regions of Interest (RoIs) and
    returns an integer ``DataArray`` where ``+1`` marks an entry event
    and ``-1`` marks an exit event.

    Parameters
    ----------
    data : xarray.DataArray
        Position data to check against the ``regions``. Must contain at
        least ``time`` and ``space`` dimensions. If a ``keypoints``
        dimension is present, it is collapsed according to ``mode``.
    regions : Sequence[BaseRegionOfInterest]
        Regions of Interest to detect entries and exits for.
    mode : {"centroid", "all", "any", "majority"}, optional
        How to aggregate across the ``keypoints`` dimension when present.

        - ``"centroid"``: compute the spatial mean across keypoints
          first, then check occupancy of the resulting centroid point.
        - ``"all"``: the subject is inside only if **all** keypoints
          are inside the region.
        - ``"any"``: the subject is inside if **any** keypoint is
          inside the region.
        - ``"majority"``: the subject is inside if **more than half**
          of the keypoints are inside the region.

        If no ``keypoints`` dimension is present, ``mode`` has no
        effect. Default is ``"centroid"``.
    min_frames : int, optional
        Minimum number of consecutive frames a subject must remain in
        the new state (inside or outside) for the transition to be
        registered. Transitions that last fewer than ``min_frames``
        frames are treated as noise and suppressed. Default is ``1``
        (no filtering).

    Returns
    -------
    xarray.DataArray
        An integer ``DataArray`` with the same dimensions as the output
        of :func:`compute_region_occupancy` — i.e. ``space`` and
        ``keypoints`` are removed, and ``region`` is added. Values:

        - ``+1`` at time points where an entry event occurred.
        - ``-1`` at time points where an exit event occurred.
        - ``0`` at all other time points.

        At ``time=0``, the value is ``+1`` if the subject starts
        inside the region, and ``0`` otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from movement.roi import PolygonOfInterest, compute_entry_exits
    >>> square = PolygonOfInterest(
    ...     [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    ...     name="square",
    ... )
    >>> positions = xr.DataArray(
    ...     np.array([[1.5, 1.5], [0.5, 0.5], [0.5, 0.5], [1.5, 1.5]]),
    ...     dims=["time", "space"],
    ...     coords={"space": ["x", "y"], "time": [0, 1, 2, 3]},
    ... )
    >>> events = compute_entry_exits(positions, [square])
    >>> events.sel(region="square").values
    array([ 0,  1,  0, -1])

    Notes
    -----
    The ``min_frames`` parameter uses a debounce approach: a state
    change (entry or exit) is only registered after the new state has
    been sustained for ``min_frames`` consecutive frames.  Both brief
    entries and brief exits are suppressed symmetrically.  The
    subject's state at ``time=0`` is always accepted as-is, so a
    ``+1`` entry is correctly recorded at the first frame when the
    subject starts inside a region.

    """
    valid_modes = {"centroid", "all", "any", "majority"}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode {mode!r}. Expected one of {sorted(valid_modes)}."
        )
    if min_frames < 1:
        raise ValueError(f"min_frames must be >= 1, got {min_frames}.")

    # Collapse keypoints via spatial centroid before computing occupancy
    if mode == "centroid" and "keypoints" in data.dims:
        position_data = data.mean(dim="keypoints")
    else:
        position_data = data

    # Compute per-region boolean occupancy
    occupancy = compute_region_occupancy(position_data, regions)

    # Reduce keypoints dimension for non-centroid modes
    if "keypoints" in occupancy.dims:
        if mode == "all":
            occupancy = occupancy.all(dim="keypoints")
        elif mode == "any":
            occupancy = occupancy.any(dim="keypoints")
        elif mode == "majority":
            n_kp = occupancy.sizes["keypoints"]
            occupancy = occupancy.sum(dim="keypoints") > (n_kp / 2)

    # Suppress brief border noise: debounce state changes so that a new
    # state (inside or outside) must be sustained for ``min_frames``
    # consecutive frames before it takes effect.  Applies symmetrically
    # to both entries and exits; time=0 is always accepted as-is.
    if min_frames > 1:
        time_axis = occupancy.get_axis_num("time")
        occ_np = occupancy.values
        occ_time_first = np.moveaxis(occ_np, time_axis, 0)
        T = occ_time_first.shape[0]
        if T > 0:
            flat = occ_time_first.reshape(T, -1)
            debounced = flat.copy()
            for col in range(flat.shape[1]):
                state = bool(flat[0, col])
                debounced[0, col] = state
                diff_run = 0
                for t in range(1, T):
                    v = bool(flat[t, col])
                    if v == state:
                        diff_run = 0
                        debounced[t, col] = state
                    else:
                        diff_run += 1
                        if diff_run >= min_frames:
                            state = v
                            debounced[t, col] = state
                            diff_run = 0
                        else:
                            debounced[t, col] = state
            debounced = debounced.reshape(occ_time_first.shape)
            debounced = np.moveaxis(debounced, 0, time_axis)
            occupancy = xr.DataArray(
                debounced.astype(bool),
                coords=occupancy.coords,
                dims=occupancy.dims,
            )

    # Compute transitions: diff gives +1 (entry) or -1 (exit)
    # diff removes the first time point; we prepend t=0 separately
    transitions = occupancy.astype(int).diff(dim="time")

    # At t=0: +1 if the subject starts inside, 0 otherwise
    initial = occupancy.isel(time=0).astype(int).expand_dims("time")

    result = xr.concat([initial, transitions], dim="time")

    # Ensure region is the leading dimension, consistent with
    # compute_region_occupancy
    dims_order = ["region"] + [d for d in result.dims if d != "region"]
    return result.transpose(*dims_order)
