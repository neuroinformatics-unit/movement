"""Compute kinematic variables like velocity and acceleration."""

import xarray as xr

from movement.utils.logging import log_error


def compute_displacement(data: xr.DataArray) -> xr.DataArray:
    """Compute displacement array in cartesian coordinates.

    The displacement array is defined as the difference between the position
    array at time point ``t`` and the position array at time point ``t-1``.

    As a result, for a given individual and keypoint, the displacement vector
    at time point ``t``, is the vector pointing from the previous
    ``(t-1)`` to the current ``(t)`` position, in cartesian coordinates.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array containing position vectors in cartesian
        coordinates, with ``time`` as a dimension.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing displacement vectors in cartesian
        coordinates.

    Notes
    -----
    For the ``position`` array of a ``poses`` dataset, the ``displacement``
    array will hold the displacement vectors for every keypoint and every
    individual.

    For the ``position`` array of a ``bboxes`` dataset, the ``displacement``
    array will hold the displacement vectors for the centroid of every
    individual bounding box.

    For the ``shape`` array of a ``bboxes`` dataset, the
    ``displacement`` array will hold vectors with the change in width and
    height per bounding box, between consecutive time points.

    """
    _validate_time_dimension(data)
    result = data.diff(dim="time")
    result = result.reindex(data.coords, fill_value=0)
    return result


def compute_velocity(data: xr.DataArray) -> xr.DataArray:
    """Compute velocity array in cartesian coordinates.

    The velocity array is the first time derivative of the position
    array.

    It is computed by applying the second order accurate central
    differences method on the position array.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array containing position vectors in cartesian
        coordinates, with ``time`` as a dimension.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing velocity vectors in cartesian
        coordinates.

    Notes
    -----
    For the ``position`` array of a ``poses`` dataset, the ``velocity`` array
    will hold the velocity vectors for every keypoint and every individual.

    For the ``position`` array of a ``bboxes`` dataset, the ``velocity`` array
    will hold the velocity vectors for the centroid of every individual
    bounding box.

    See Also
    --------
    :py:meth:`xarray.DataArray.differentiate` : The underlying method used.

    """
    return _compute_approximate_time_derivative(data, order=1)


def compute_acceleration(data: xr.DataArray) -> xr.DataArray:
    """Compute acceleration array in cartesian coordinates.

    The acceleration array is the second time derivative of the
    position array.

    It is computed by applying the second order accurate central
    differences method on the velocity array.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array containing position vectors in cartesian
        coordinates, with``time`` as a dimension.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing acceleration vectors in cartesian
        coordinates.

    Notes
    -----
    For the ``position`` array of a ``poses`` dataset, the ``acceleration``
    array will hold the acceleration vectors for every keypoint and every
    individual.

    For the ``position`` array of a ``bboxes`` dataset, the ``acceleration``
    array will hold the acceleration vectors for the centroid of every
    individual bounding box.

    See Also
    --------
    :py:meth:`xarray.DataArray.differentiate` : The underlying method used.

    """
    return _compute_approximate_time_derivative(data, order=2)


def _compute_approximate_time_derivative(
    data: xr.DataArray, order: int
) -> xr.DataArray:
    """Compute the time derivative of an array using numerical differentiation.

    This function uses :py:meth:`xarray.DataArray.differentiate`,
    which differentiates the array with the second order
    accurate central differences method.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array containing ``time`` as a dimension.
    order : int
        The order of the time derivative. For an input containing position data, use
        1 to compute velocity, and 2 to compute acceleration. Value must be a
        positive integer.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the time derivative of the
        input data.

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
