"""Compute kinematic variables like velocity and acceleration."""

import itertools
from typing import Literal

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
    data: xr.DataArray, pairs: dict[str, str | list[str]] | None = None
) -> xr.DataArray | dict[str, xr.DataArray]:
    """Compute interindividual distances for all pairs of keypoints.

    Interindividual distances are computed as the norm of the
    difference in position (for each pair of the same keypoints)
    between pairs of individuals at each time point.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing
        ``('time', 'individuals', 'keypoints', 'space'`` as dimensions.
    pairs : dict[str, str | list[str]], optional
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

    See Also
    --------
    compute_interkeypoint_distances : Compute distances between pairs of
        keypoints within each individual.
    movement.utils.vector.compute_norm : Compute the norm of a vector.

    """
    return _compute_pairwise_distances(data, "individuals", pairs=pairs)


def compute_interkeypoint_distances(
    data: xr.DataArray, pairs: dict[str, str | list[str]] | None = None
) -> xr.DataArray | dict[str, xr.DataArray]:
    """Compute interkeypoint distances for all pairs of individuals.

    Interkeypoint distances are computed as the norm of the
    difference in position (for all individuals) between pairs of
    keypoints at each time point.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing
        ``('time', 'individuals', 'keypoints', 'space'`` as dimensions.
    pairs : dict[str, str | list[str]], optional
        A dictionary containing the mapping between pairs of keypoints.
        The key is the keypoint and the value is a list of keypoints
        or a string representing a keypoint to compute the distance with.
        If not provided, defaults to ``None`` and all possible combinations
        of pairs are computed.

    Returns
    -------
    xarray.DataArray | dict[str, xarray.DataArray]
        An :class:`xarray.DataArray` containing the computed distances for
        all individuals between the given pair of keypoints, or if
        multiple pairs are provided, a dictionary containing the computed
        distances for each pair of keypoints, with the key being the pair
        of keypoints and the value being the :class:`xarray.DataArray`
        containing the computed distances.

    See Also
    --------
    compute_interindividual_distances: Compute distances for each keypoint,
        between pairs of individuals.

    """
    return _compute_pairwise_distances(data, "keypoints", pairs=pairs)


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


def _compute_pairwise_distances(
    data: xr.DataArray,
    dim: Literal["individuals", "keypoints"],
    pairs: dict[str, str | list[str]] | None = None,
) -> xr.DataArray | dict[str, xr.DataArray]:
    """Compute pairwise distances between ``individuals`` or ``keypoints``.

    This function computes the distances between pairs of ``keypoints``
    (i.e. interkeypoint distances) or pairs of ``individuals`` (i.e.
    interindividual distances). The distances are computed as the norm
    of the difference in position between pairs of ``keypoints`` or
    ``individuals`` at each time point.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing
        ``('time', 'individuals', 'keypoints', 'space'`` as dimensions.
    dim : str
        The dimension to compute the distances for. Must be either
        ``'individuals'`` or ``'keypoints'``.
    pairs : dict[str, str | list[str]], optional
        A dictionary containing the mapping between pairs of ``individuals``
        or ``keypoints``. The key is the keypoint or individual and the
        value is a list of keypoints or individuals or a string
        representing a keypoint or individual to compute the distance with.
        If not provided, defaults to ``None`` and all possible combinations
        of pairs are computed.

    Returns
    -------
    xarray.DataArray | dict[str, xarray.DataArray]
        An :class:`xarray.DataArray` containing the computed distances between
        a single pair of ``individuals`` or ``keypoints``. If multiple
        ``pairs`` are provided, this will be a dictionary containing the
        computed distances for each pair, with the key being the pair of
        keypoints or individuals and the value being the
        :class:`xarray.DataArray` containing the computed distances.

    See Also
    --------
    movement.utils.vector.compute_norm : Compute the norm of a vector.

    """
    if dim not in ["individuals", "keypoints"]:
        raise log_error(
            ValueError,
            "Dimension must be either 'individuals' or 'keypoints', "
            f"but got {dim}.",
        )
    pairwise_distances = {}
    # Compute all possible pair combinations if not provided
    if pairs is None:
        paired_elements = list(
            itertools.combinations(getattr(data, dim).values, 2)
        )
    else:
        paired_elements = [
            (elem1, elem2)
            for elem1, elem2_list in pairs.items()
            for elem2 in (
                [elem2_list] if isinstance(elem2_list, str) else elem2_list
            )
        ]
    for elem1, elem2 in paired_elements:
        distance = compute_norm(
            data.sel({dim: elem1}) - data.sel({dim: elem2})
        )
        if dim in distance.coords:
            distance = distance.drop_vars(dim)
        pairwise_distances[f"dist_{elem1}_{elem2}"] = distance
    # Return DataArray if result only has one key
    if len(pairwise_distances) == 1:
        return next(iter(pairwise_distances.values()))
    return pairwise_distances
