"""Compute kinematic variables like velocity and acceleration."""

import itertools

import xarray as xr

from movement.utils.logging import log_error
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


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


def compute_interindividual_distances(
    data: xr.DataArray, pairs_mapping: dict[str, str | list[str]] | None = None
) -> xr.DataArray | dict[str, xr.DataArray]:
    """Compute interindividual distances for all pairs of keypoints.

    Interindividual distances are computed as the norm of the
    difference in position (for all keypoints) between pairs of
    individuals at each time point.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing
        ``('time', 'individuals', 'keypoints', 'space'`` as dimensions.
    pairs_mapping : dict[str, str | list[str]], optional
        A dictionary containing the mapping between pairs of individuals.
        The key is the individual and the value is a list of individuals
        or a string representing an individual to compute the distance with.
        If not provided, defaults to ``None`` and all possible combinations
        of pairs are computed.

    Returns
    -------
    xarray.DataArray | dict[str, xarray.DataArray]
        An :class:`xarray.DataArray` containing the computed distances for
        all keypoints between the given pair of individuals, or if
        multiple pairs are provided, a dictionary containing the computed
        distances for each pair of individuals, with the key being the pair
        of individuals and the value being the :class:`xarray.DataArray`
        containing the computed distances.

    """
    interindividual_distances = {}
    # Compute all possible pair combinations if not provided
    pairs = (
        list(itertools.combinations(data.individuals.values, 2))
        if pairs_mapping is None
        else [
            (ind, ind2)
            for ind, ind2_list in pairs_mapping.items()
            for ind2 in (
                ind2_list if isinstance(ind2_list, list) else [ind2_list]
            )
        ]
    )
    for ind, ind2 in pairs:
        distance = compute_norm(
            data.sel(individuals=ind) - data.sel(individuals=ind2)
        )
        if "individuals" in distance.coords:
            distance = distance.drop_vars("individuals")
        interindividual_distances[f"dist_{ind}_{ind2}"] = distance
    # Return DataArray if result only has one key
    if len(interindividual_distances) == 1:
        return next(iter(interindividual_distances.values()))
    return interindividual_distances


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
