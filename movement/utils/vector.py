"""Utility functions for vector operations."""

import numpy as np
import xarray as xr

from movement.utils.logging import log_error
from movement.validators.arrays import validate_dims_coords


def compute_norm(data: xr.DataArray) -> xr.DataArray:
    """Compute the norm of the vectors along the spatial dimension.

    The norm of a vector is its magnitude, also called Euclidean norm, 2-norm
    or Euclidean length. Note that if the input data is expressed in polar
    coordinates, the magnitude of a vector is the same as its radial coordinate
    ``rho``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array containing either ``space`` or ``space_pol``
        as a dimension.

    Returns
    -------
    xarray.DataArray
         A data array holding the norm of the input vectors.
         Note that this output array has no spatial dimension but preserves
         all other dimensions of the input data array (see Notes).

    Notes
    -----
    If the input data array is a ``position`` array, this function will compute
    the magnitude of the position vectors, for every individual and keypoint,
    at every timestep. If the input data array is a ``shape`` array of a
    bounding boxes dataset, it will compute the magnitude of the shape
    vectors (i.e., the diagonal of the bounding box),
    for every individual and at every timestep.


    """
    if "space" in data.dims:
        validate_dims_coords(data, {"space": ["x", "y"]})
        return xr.apply_ufunc(
            np.linalg.norm,
            data,
            input_core_dims=[["space"]],
            kwargs={"axis": -1},
        )
    elif "space_pol" in data.dims:
        validate_dims_coords(data, {"space_pol": ["rho", "phi"]})
        return data.sel(space_pol="rho", drop=True)
    else:
        _raise_error_for_missing_spatial_dim()


def convert_to_unit(data: xr.DataArray) -> xr.DataArray:
    """Convert the vectors along the spatial dimension into unit vectors.

    A unit vector is a vector pointing in the same direction as the original
    vector but with norm = 1.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array containing either ``space`` or ``space_pol``
        as a dimension.

    Returns
    -------
    xarray.DataArray
        A data array holding the unit vectors of the input data array
        (all input dimensions are preserved).

    Notes
    -----
    Note that the unit vector for the null vector is undefined, since the null
    vector has 0 norm and no direction associated with it.

    """
    if "space" in data.dims:
        validate_dims_coords(data, {"space": ["x", "y"]})
        return data / compute_norm(data)
    elif "space_pol" in data.dims:
        validate_dims_coords(data, {"space_pol": ["rho", "phi"]})
        # Set both rho and phi values to NaN at null vectors (where rho = 0)
        new_data = xr.where(data.sel(space_pol="rho") == 0, np.nan, data)
        # Set the rho values to 1 for non-null vectors (phi is preserved)
        new_data.loc[{"space_pol": "rho"}] = xr.where(
            new_data.sel(space_pol="rho").isnull(), np.nan, 1
        )
        return new_data
    else:
        _raise_error_for_missing_spatial_dim()


def cart2pol(data: xr.DataArray) -> xr.DataArray:
    """Transform Cartesian coordinates to polar.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``space`` as a dimension,
        with ``x`` and ``y`` in the dimension coordinate.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the polar coordinates
        stored in the ``space_pol`` dimension, with ``rho``
        and ``phi`` in the dimension coordinate. The angles
        ``phi`` returned are in radians, in the range ``[-pi, pi]``.

    """
    validate_dims_coords(data, {"space": ["x", "y"]})
    rho = compute_norm(data)
    phi = xr.apply_ufunc(
        np.arctan2,
        data.sel(space="y"),
        data.sel(space="x"),
    )
    # Replace space dim with space_pol
    dims = list(data.dims)
    dims[dims.index("space")] = "space_pol"
    return xr.combine_nested(
        [
            rho.assign_coords({"space_pol": "rho"}),
            phi.assign_coords({"space_pol": "phi"}),
        ],
        concat_dim="space_pol",
    ).transpose(*dims)


def pol2cart(data: xr.DataArray) -> xr.DataArray:
    """Transform polar coordinates to Cartesian.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``space_pol`` as a dimension,
        with ``rho`` and ``phi`` in the dimension coordinate.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the Cartesian coordinates
        stored in the ``space`` dimension, with ``x`` and ``y``
        in the dimension coordinate.

    """
    validate_dims_coords(data, {"space_pol": ["rho", "phi"]})
    rho = data.sel(space_pol="rho")
    phi = data.sel(space_pol="phi")
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    # Replace space_pol dim with space
    dims = list(data.dims)
    dims[dims.index("space_pol")] = "space"
    return xr.combine_nested(
        [
            x.assign_coords({"space": "x"}),
            y.assign_coords({"space": "y"}),
        ],
        concat_dim="space",
    ).transpose(*dims)


def _raise_error_for_missing_spatial_dim() -> None:
    raise log_error(
        ValueError,
        "Input data array must contain either 'space' or 'space_pol' "
        "as dimensions.",
    )
