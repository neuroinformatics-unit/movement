"""Utility functions for vector operations."""

import numpy as np
import xarray as xr

from movement.utils.logging import log_error


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
    _validate_dimension_coordinates(data, {"space": ["x", "y"]})
    rho = magnitude(data)
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
    _validate_dimension_coordinates(data, {"space_pol": ["rho", "phi"]})
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


def magnitude(data: xr.DataArray) -> xr.DataArray:
    """Compute the magnitude in space.

    The magnitude is computed as the Euclidean norm of a vector
    with spatial components ``x`` and ``y`` in Cartesian coordinates.
    If the input data contains polar coordinates, the magnitude
    is the same as the radial distance ``rho``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing either ``space`` or ``space_pol``
        as a dimension.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the magnitude of the vector
        in space. The output has no spatial dimension.


    """
    if "space" in data.dims:
        _validate_dimension_coordinates(data, {"space": ["x", "y"]})
        return xr.apply_ufunc(
            np.linalg.norm,
            data,
            input_core_dims=[["space"]],
            kwargs={"axis": -1},
        )
    elif "space_pol" in data.dims:
        _validate_dimension_coordinates(data, {"space_pol": ["rho", "phi"]})
        return data.sel(space_pol="rho", drop=True)
    else:
        raise log_error(
            ValueError,
            "Input data must contain either 'space' or 'space_pol' "
            "as dimensions.",
        )


def normalize(data: xr.DataArray) -> xr.DataArray:
    """Normalize data by the magnitude in space.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``space`` as a dimension,
        with ``x`` and ``y`` in the dimension coordinate.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the normalized data,
        having the same dimensions as the input data.

    Notes
    -----
    Where the input values are 0 for both ``x`` and ``y``, the normalized
    values will be NaN, because of zero-division.

    """
    _validate_dimension_coordinates(data, {"space": ["x", "y"]})
    return data / magnitude(data)


def _validate_dimension_coordinates(
    data: xr.DataArray, required_dim_coords: dict
) -> None:
    """Validate the input data array.

    Ensure that it contains the required dimensions and coordinates.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to validate.
    required_dim_coords : dict
        A dictionary of required dimensions and their corresponding
        coordinate values.

    Raises
    ------
    ValueError
        If the input data does not contain the required dimension(s)
        and/or the required coordinate(s).

    """
    missing_dims = [dim for dim in required_dim_coords if dim not in data.dims]
    error_message = ""
    if missing_dims:
        error_message += (
            f"Input data must contain {missing_dims} as dimensions.\n"
        )
    missing_coords = []
    for dim, coords in required_dim_coords.items():
        missing_coords = [
            coord for coord in coords if coord not in data.coords.get(dim, [])
        ]
        if missing_coords:
            error_message += (
                "Input data must contain "
                f"{missing_coords} in the '{dim}' coordinates."
            )
    if error_message:
        raise log_error(ValueError, error_message)
