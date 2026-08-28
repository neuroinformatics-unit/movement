"""Angle computation utilities for joint and geometric analysis.

This module provides xarray-native functions for computing angles at a vertex
using three points, leveraging vector operations from movement.utils.vector.


Main feature:
    - compute_vertex_angle: Compute the angle at a specified vertex using
      the dot product.

See Also:
    - movement.utils.vector for vector normalization and dot product utilities.

"""

import numpy as np
import xarray as xr

from movement.utils.vector import convert_to_unit


def compute_vertex_angle(point1, vertex, point3):
    """Calculate the joint angle at a vertex given three points (per frame).

    Uses the dot product for computation.

    This function uses movement.utils.vector utilities for normalization.

    Parameters
    ----------
    point1 : np.ndarray or xr.DataArray
        First point(s), shape (..., 2)
    vertex : np.ndarray or xr.DataArray
        Vertex point(s), shape (..., 2)
    point3 : np.ndarray or xr.DataArray
        Third point(s), shape (..., 2)

    Returns
    -------
    np.ndarray or xr.DataArray
        Angle at the vertex in radians, shape (...,)

    """

    # Convert to xarray.DataArray for compatibility with vector utilities
    def to_xr(arr):
        if isinstance(arr, xr.DataArray):
            return arr
        arr = np.asarray(arr)
        if arr.shape[-1] == 2:
            return xr.DataArray(
                arr, dims=["space"], coords={"space": ["x", "y"]}
            )
        return xr.DataArray(arr)

    p1 = to_xr(point1)
    vtx = to_xr(vertex)
    p3 = to_xr(point3)
    v1 = p1 - vtx
    v2 = p3 - vtx
    v1_unit = convert_to_unit(v1)
    v2_unit = convert_to_unit(v2)
    dot = (
        (v1_unit * v2_unit).sum(dim="space")
        if isinstance(v1_unit, xr.DataArray)
        else np.sum(v1_unit * v2_unit, axis=-1)
    )
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)
    return angle
