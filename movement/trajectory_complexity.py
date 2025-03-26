"""Compute trajectory complexity measures.

This module provides functions to compute various measures of trajectory 
complexity, which quantify how straight or tortuous a path is. These metrics 
are useful for analyzing animal movement patterns across space.
"""

import numpy as np
import xarray as xr

from movement.kinematics import compute_displacement, compute_path_length
from movement.utils.logging import log_error, log_to_attrs, log_warning
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


@log_to_attrs
def compute_straightness_index(
    data: xr.DataArray,
    start: float | None = None,
    stop: float | None = None,
) -> xr.DataArray:
    """Compute the straightness index of a trajectory.
    
    The straightness index is defined as the ratio of the Euclidean distance 
    between the start and end points of a trajectory to the total path length. 
    Values range from 0 to 1, where 1 indicates a perfectly straight path,
    and values closer to 0 indicate more tortuous paths.
    
    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    start : float, optional
        The start time of the trajectory. If None (default),
        the minimum time coordinate in the data is used.
    stop : float, optional
        The end time of the trajectory. If None (default),
        the maximum time coordinate in the data is used.
        
    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed straightness index,
        with dimensions matching those of the input data,
        except ``time`` and ``space`` are removed.
        
    Notes
    -----
    The straightness index (SI) is calculated as:
    
    SI = Euclidean distance / Path length
    
    where the Euclidean distance is the straight-line distance between the
    start and end points, and the path length is the total distance traveled
    along the trajectory.
    
    References
    ----------
    .. [1] Batschelet, E. (1981). Circular statistics in biology.
           London: Academic Press.
    """
    validate_dims_coords(data, {"time": [], "space": []})

    # Determine start and stop points
    if start is None:
        start = data.time.min().item()
    if stop is None:
        stop = data.time.max().item()

    # Extract start and end positions
    start_pos = data.sel(time=start, method="nearest")
    end_pos = data.sel(time=stop, method="nearest")

    # Calculate Euclidean distance between start and end points
    euclidean_distance = compute_norm(end_pos - start_pos)

    # Calculate path length
    path_length = compute_path_length(data, start=start, stop=stop)

    # Compute straightness index
    straightness_index = euclidean_distance / path_length

    return straightness_index 