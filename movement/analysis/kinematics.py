import numpy as np
import xarray as xr

from movement.logging import log_error


def compute_displacement(data: xr.DataArray) -> xr.DataArray:
    """Compute the displacement between consecutive positions
    of each keypoint for each individual across time.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``time`` as a dimension.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed displacement.
    """
    _validate_time_dimension(data)
    result = data.diff(dim="time")
    result = result.reindex(data.coords, fill_value=0)
    return result


def compute_velocity(data: xr.DataArray) -> xr.DataArray:
    """Compute the velocity between consecutive positions
    of each keypoint for each individual across time.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``time`` as a dimension.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed velocity.

    Notes
    -----
    This function computes velocity using numerical differentiation
    and assumes equidistant time spacing.
    """
    return _compute_approximate_derivative(data, order=1)


def compute_acceleration(data: xr.DataArray) -> xr.DataArray:
    """Compute the acceleration between consecutive positions
    of each keypoint for each individual across time.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``time`` as a dimension.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed acceleration.

    Notes
    -----
    This function computes acceleration using numerical differentiation
    and assumes equidistant time spacing.
    """
    return _compute_approximate_derivative(data, order=2)


def _compute_approximate_derivative(
    data: xr.DataArray, order: int
) -> xr.DataArray:
    """Compute velocity or acceleration using numerical differentiation,
    assuming equidistant time spacing.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``time`` as a dimension.
    order : int
        The order of the derivative. 1 for velocity, 2 for
        acceleration. Value must be a positive integer.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the derived variable.
    """
    if not isinstance(order, int):
        raise log_error(
            TypeError, f"Order must be an integer, but got {type(order)}."
        )
    if order <= 0:
        raise log_error(ValueError, "Order must be a positive integer.")
    _validate_time_dimension(data)
    result = data
    dt = data["time"].values[1] - data["time"].values[0]
    for _ in range(order):
        result = xr.apply_ufunc(
            np.gradient,
            result,
            dt,
            kwargs={"axis": 0},
        )
    result = result.reindex_like(data)
    return result


def _validate_time_dimension(data: xr.DataArray) -> None:
    """Validate the input data contains a ``time`` dimension.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to validate.

    Raises
    ------
    ValueError
        If the input data does not contain a ``time`` dimension.
    """
    if "time" not in data.dims:
        raise log_error(
            ValueError, "Input data must contain 'time' as a dimension."
        )
