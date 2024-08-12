"""Compute kinematic variables like velocity and acceleration."""

import xarray as xr

from movement.utils.logging import log_error


def compute_displacement(data: xr.DataArray) -> xr.DataArray:
    """Compute displacement between consecutive positions.

    Displacement is the difference between consecutive positions
    of each keypoint for each individual across ``time``.
    At each time point ``t``, it is defined as a vector in
    cartesian ``(x,y)`` coordinates, pointing from the previous
    ``(t-1)`` to the current ``(t)`` position.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with
        ``time`` as a dimension.

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
    """Compute the velocity in cartesian ``(x,y)`` coordinates.

    Velocity is the first derivative of position for each keypoint
    and individual across ``time``, computed with the second order
    accurate central differences.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with
        ``time`` as a dimension.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed velocity.

    See Also
    --------
    :py:meth:`xarray.DataArray.differentiate` : The underlying method used.

    """
    return _compute_approximate_time_derivative(data, order=1)


def compute_acceleration(data: xr.DataArray) -> xr.DataArray:
    """Compute acceleration in cartesian ``(x,y)`` coordinates.

    Acceleration is the second derivative of position for each keypoint
    and individual across ``time``, computed with the second order
    accurate central differences.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with
        ``time`` as a dimension.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed acceleration.

    See Also
    --------
    :py:meth:`xarray.DataArray.differentiate` : The underlying method used.

    """
    return _compute_approximate_time_derivative(data, order=2)


def _compute_approximate_time_derivative(
    data: xr.DataArray, order: int
) -> xr.DataArray:
    """Compute the derivative using numerical differentiation.

    This function uses :py:meth:`xarray.DataArray.differentiate`,
    which differentiates the array with the second order
    accurate central differences.

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
    for _ in range(order):
        result = result.differentiate("time")
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
