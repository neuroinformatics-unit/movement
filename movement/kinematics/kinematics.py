"""Compute core kinematic variables such as time derivatives of position.

This module provides functions for computing fundamental kinematic properties
such as forward & backward displacement, velocity, acceleration, and speed.
The ``movement.kinematics`` subpackage encompasses a broader range of
functionality (e.g., orientations and distances ), but this file is
intended to isolate 'true' kinematics for clarity. In a future release, the
public API may be revised to reflect this distinction more explicitly.

"""

import warnings

import xarray as xr

from movement.utils.logging import logger
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


def compute_time_derivative(data: xr.DataArray, order: int) -> xr.DataArray:
    """Compute the time-derivative of an array using numerical differentiation.

    This function uses :meth:`xarray.DataArray.differentiate`,
    which differentiates the array with the second-order
    accurate central differences method.

    Parameters
    ----------
    data
        The input data containing ``time`` as a required dimension.
    order
        The order of the time-derivative. For an input containing position
        data, use 1 to compute velocity, and 2 to compute acceleration. Value
        must be a positive integer.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the time-derivative of the input data.

    See Also
    --------
    :meth:`xarray.DataArray.differentiate` : The underlying method used.

    """
    if not isinstance(order, int):
        raise logger.error(
            TypeError(f"Order must be an integer, but got {type(order)}.")
        )
    if order <= 0:
        raise logger.error(ValueError("Order must be a positive integer."))
    validate_dims_coords(data, {"time": []})
    result = data
    for _ in range(order):
        result = result.differentiate("time")
    return result


def compute_displacement(data: xr.DataArray) -> xr.DataArray:
    """Compute displacement array in Cartesian coordinates.

    .. deprecated:: 0.9.1
        This function is deprecated and will be removed in a future release.
        Use :func:`compute_forward_displacement` or
        :func:`compute_backward_displacement` instead.

    The displacement array is defined as the difference between the position
    array at time point ``t`` and the position array at time point ``t-1``.

    As a result, for a given individual and keypoint, the displacement vector
    at time point ``t``, is the vector pointing from the previous
    ``(t-1)`` to the current ``(t)`` position, in Cartesian coordinates.

    Parameters
    ----------
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing displacement vectors in Cartesian
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
    warnings.warn(
        "The function `movement.kinematics.compute_displacement` is deprecated"
        " and will be removed in a future release. "
        "Please use `movement.kinematics.compute_forward_displacement` or "
        "`movement.kinematics.compute_backward_displacement` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    validate_dims_coords(data, {"time": [], "space": []})
    result = data.diff(dim="time")
    result = result.reindex_like(data, fill_value=0)
    result.name = "displacement"
    return result


def _compute_forward_displacement(data: xr.DataArray) -> xr.DataArray:
    """Compute forward displacement vectors in Cartesian coordinates.

    The displacement vectors have origin at the position at time t,
    pointing to the position at time t+1.
    The last vector is of magnitude=0.
    """
    validate_dims_coords(data, {"time": [], "space": []})
    result = data.diff(dim="time", label="lower")
    result = result.reindex_like(data, fill_value=0)
    return result


def compute_forward_displacement(data: xr.DataArray) -> xr.DataArray:
    """Compute forward displacement array in Cartesian coordinates.

    The forward displacement array is defined as the difference between the
    position array at time point ``t+1`` and the position array at time point
    ``t``.

    As a result, for a given individual and keypoint, the forward displacement
    vector at time point ``t``, is the vector pointing from the current ``t``
    position to the next ``t+1``, in Cartesian coordinates.

    Parameters
    ----------
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing forward displacement vectors in
        Cartesian coordinates.

    Notes
    -----
    For the ``position`` array of a ``poses`` dataset, the
    ``forward_displacement`` array will hold the forward displacement vectors
    for every keypoint and every individual.

    For the ``position`` array of a ``bboxes`` dataset, the
    ``forward_displacement`` array will hold the forward displacement vectors
    for the centroid of every individual bounding box.

    For the ``shape`` array of a ``bboxes`` dataset, the
    ``forward_displacement`` array will hold vectors with the change in width
    and height per bounding box, between consecutive time points.

    """
    result = _compute_forward_displacement(data)
    result.name = "forward_displacement"
    return result


def compute_backward_displacement(data: xr.DataArray) -> xr.DataArray:
    """Compute backward displacement array in Cartesian coordinates.

    The backward displacement array is defined as the difference between the
    position array at time point ``t-1`` and the position array at time point
    ``t``.

    As a result, for a given individual and keypoint, the backward displacement
    vector at time point ``t``, is the vector pointing from the current ``t``
    position to the previous ``t-1`` in Cartesian coordinates.

    Parameters
    ----------
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing backward displacement vectors in
        Cartesian coordinates.

    Notes
    -----
    For the ``position`` array of a ``poses`` dataset, the
    ``backward_displacement`` array will hold the backward displacement vectors
    for every keypoint and every individual.

    For the ``position`` array of a ``bboxes`` dataset, the
    ``backward_displacement`` array will hold the backward displacement vectors
    for the centroid of every individual bounding box.

    For the ``shape`` array of a ``bboxes`` dataset, the
    ``backward_displacement`` array will hold vectors with the change in width
    and height per bounding box, between consecutive time points.

    """
    fwd_displacement = _compute_forward_displacement(data)
    backward_displacement = -fwd_displacement.roll(time=1)
    backward_displacement.name = "backward_displacement"
    return backward_displacement


def compute_velocity(data: xr.DataArray) -> xr.DataArray:
    """Compute velocity array in Cartesian coordinates.

    The velocity array is the first time-derivative of the position
    array. It is computed by applying the second-order accurate central
    differences method on the position array.

    Parameters
    ----------
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing velocity vectors in Cartesian
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
    compute_time_derivative : The underlying function used.

    """
    # validate only presence of Cartesian space dimension
    # (presence of time dimension will be checked in compute_time_derivative)
    validate_dims_coords(data, {"space": []})
    result = compute_time_derivative(data, order=1)
    result.name = "velocity"
    return result


def compute_acceleration(data: xr.DataArray) -> xr.DataArray:
    """Compute acceleration array in Cartesian coordinates.

    The acceleration array is the second time-derivative of the
    position array. It is computed by applying the second-order accurate
    central differences method on the velocity array.

    Parameters
    ----------
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing acceleration vectors in Cartesian
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
    compute_time_derivative : The underlying function used.

    """
    # validate only presence of Cartesian space dimension
    # (presence of time dimension will be checked in compute_time_derivative)
    validate_dims_coords(data, {"space": []})
    result = compute_time_derivative(data, order=2)
    result.name = "acceleration"
    return result


def compute_speed(data: xr.DataArray) -> xr.DataArray:
    """Compute instantaneous speed at each time point.

    Speed is a scalar quantity computed as the Euclidean norm (magnitude)
    of the velocity vector at each time point.


    Parameters
    ----------
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed speed,
        with dimensions matching those of the input data,
        except ``space`` is removed.

    """
    result = compute_norm(compute_velocity(data))
    result.name = "speed"
    return result

