"""Compute kinematic variables like velocity and acceleration."""

import numpy as np
import xarray as xr

from movement.utils.logging import log_error
from movement.validators.arrays import validate_dims_coords
from movement.utils.vector import convert_to_unit


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
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

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
    validate_dims_coords(data, {"time": [], "space": []})
    result = data.diff(dim="time")
    result = result.reindex(data.coords, fill_value=0)
    return result


def compute_velocity(data: xr.DataArray) -> xr.DataArray:
    """Compute velocity array in cartesian coordinates.

    The velocity array is the first time-derivative of the position
    array. It is computed by applying the second-order accurate central
    differences method on the position array.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

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
    compute_time_derivative : The underlying function used.

    """
    # validate only presence of Cartesian space dimension
    # (presence of time dimension will be checked in compute_time_derivative)
    validate_dims_coords(data, {"space": []})
    return compute_time_derivative(data, order=1)


def compute_acceleration(data: xr.DataArray) -> xr.DataArray:
    """Compute acceleration array in cartesian coordinates.

    The acceleration array is the second time-derivative of the
    position array. It is computed by applying the second-order accurate
    central differences method on the velocity array.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

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
    compute_time_derivative : The underlying function used.

    """
    # validate only presence of Cartesian space dimension
    # (presence of time dimension will be checked in compute_time_derivative)
    validate_dims_coords(data, {"space": []})
    return compute_time_derivative(data, order=2)


def compute_time_derivative(data: xr.DataArray, order: int) -> xr.DataArray:
    """Compute the time-derivative of an array using numerical differentiation.

    This function uses :meth:`xarray.DataArray.differentiate`,
    which differentiates the array with the second-order
    accurate central differences method.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``time`` as a required dimension.
    order : int
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
        raise log_error(
            TypeError, f"Order must be an integer, but got {type(order)}."
        )
    if order <= 0:
        raise log_error(ValueError, "Order must be a positive integer.")
    validate_dims_coords(data, {"time": []})
    result = data
    for _ in range(order):
        result = result.differentiate("time")
    return result


def compute_head_direction_vector(
    data: xr.DataArray,
    left_keypoint: str,
    right_keypoint: str,
    front_keypoint: str | None = None,
):
    """Compute the 2D head direction vector given two keypoints on the head.

    The head direction vector is computed as a vector perpendicular to the
    line connecting two keypoints on either side of the head, pointing
    forwards (in the rostral direction). As the forward direction may
    differ between coordinate systems, the front keypoint is used ...,
    when present. Otherwise, we assume that coordinates are given in the
    image coordinate system (where the origin is located in the top-left).

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two chosen keypoints corresponding to the left and
        right of the head.
    left_keypoint : str
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint : str
        Name of the right keypoint, e.g., "right_ear"
    front_keypoint : str | None
        (Optional) Name of the front keypoint, e.g., "nose".

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the head direction vector,
        with dimensions matching the input data array, but without the
        ``keypoints`` dimension.

    """
    # Validate input dataset
    _validate_type_data_array(data)
    _validate_time_keypoints_space_dimensions(data)

    if left_keypoint == right_keypoint:
        raise log_error(
            ValueError, "The left and right keypoints may not be identical."
        )
    if len(data.space) != 2:
        raise log_error(
            ValueError,
            "Input data must have 2 (and only 2) spatial dimensions, but "
            f"currently has {len(data.space)}.",
        )

    # Select the right and left keypoints
    head_left = data.sel(keypoints=left_keypoint, drop=True)
    head_right = data.sel(keypoints=right_keypoint, drop=True)

    # Initialize a vector from right to left ear, and another vector
    # perpendicular to the X-Y plane
    right_to_left_vector = head_left - head_right
    perpendicular_vector = np.array([0, 0, -1])

    # Compute cross product
    head_vector = head_right.copy()
    head_vector.values = np.cross(right_to_left_vector, perpendicular_vector)[
        :, :, :-1
    ]

    # Check computed head_vector is pointing in the same direction as vector
    # from head midpoint to snout
    if front_keypoint:
        head_front = data.sel(keypoints=front_keypoint, drop=True)
        head_midpoint = (head_right + head_left) / 2
        mid_to_front_vector = head_front - head_midpoint
        dot_product_array = (
            convert_to_unit(head_vector.sel(individuals=data.individuals[0]))
            * convert_to_unit(mid_to_front_vector).sel(
                individuals=data.individuals[0]
            )
        ).sum(dim="space")
        median_dot_product = float(dot_product_array.median(dim="time").values)
        if median_dot_product < 0:
            perpendicular_vector = np.array([0, 0, 1])
            head_vector.values = np.cross(
                right_to_left_vector, perpendicular_vector
            )[:, :, :-1]

    return head_vector


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


def _validate_type_data_array(data: xr.DataArray) -> None:
    """Validate the input data is an xarray DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to validate.

    Raises
    ------
    ValueError
        If the input data is not an xarray DataArray.

    """
    if not isinstance(data, xr.DataArray):
        raise log_error(
            TypeError,
            f"Input data must be an xarray.DataArray, but got {type(data)}.",
        )


def _validate_time_keypoints_space_dimensions(data: xr.DataArray) -> None:
    """Validate if input data contains ``time``, ``keypoints`` and ``space``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to validate.

    Raises
    ------
    ValueError
        If the input data is not an xarray DataArray.

    """
    if not all(coord in data.dims for coord in ["time", "keypoints", "space"]):
        raise log_error(
            AttributeError,
            "Input data must contain 'time', 'space', and 'keypoints' as "
            "dimensions.",
        )