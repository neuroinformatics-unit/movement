"""Compute kinematic variables like velocity and acceleration."""

import itertools
from typing import Literal

import xarray as xr
from scipy.spatial.distance import cdist

from movement.utils.logging import log_error
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


def _cdist(
    a: xr.DataArray,
    b: xr.DataArray,
    dim: Literal["individuals", "keypoints"],
    metric: str | None = "euclidean",
    **kwargs,
) -> xr.DataArray:
    """Compute distances between two position arrays across a given dimension.

    This function is a wrapper around :func:`scipy.spatial.distance.cdist`
    and computes the pairwise distances between the two input position arrays
    across the dimension specified by ``dim``.
    The dimension can be either ``individuals`` or ``keypoints``.
    The distances are computed using the specified ``metric``.

    Parameters
    ----------
    a : xarray.DataArray
        The first input data containing position information of a
        single individual or keypoint, with ``space``
        (in Cartesian coordinates) as a required dimension.
    b : xarray.DataArray
        The second input data containing position information of a
        single individual or keypoint, with ``space``
        (in Cartesian coordinates) as a required dimension.
    dim : str
        The dimension to compute the distances for. Must be either
        ``'individuals'`` or ``'keypoints'``.
    metric : str, optional
        The distance metric to use. Must be one of the options supported
        by :func:`scipy.spatial.distance.cdist`, e.g. ``'cityblock'``,
        ``'euclidean'``, etc.
        Defaults to ``'euclidean'``.
    **kwargs : dict
        Additional keyword arguments to pass to
        :func:`scipy.spatial.distance.cdist`.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed distances between
        each pair of inputs.

    Examples
    --------
    Compute the Euclidean distance (default) between ``ind1`` and
    ``ind2`` (i.e. interindividual distance for all keypoints)
    using the ``position`` data variable in the Dataset ``ds``:

    >>> pos1 = ds.position.sel(individuals="ind1")
    >>> pos2 = ds.position.sel(individuals="ind2")
    >>> ind_dists = _cdist(pos1, pos2, dim="individuals")

    Compute the Euclidean distance (default) between ``key1`` and
    ``key2`` (i.e. interkeypoint distance for all individuals)
    using the ``position`` data variable in the Dataset ``ds``:

    >>> pos1 = ds.position.sel(keypoints="key1")
    >>> pos2 = ds.position.sel(keypoints="key2")
    >>> key_dists = _cdist(pos1, pos2, dim="keypoints")

    See Also
    --------
    scipy.spatial.distance.cdist : The underlying function used.
    compute_pairwise_distances : Compute pairwise distances between
        ``individuals`` or ``keypoints``

    """
    core_dim = "individuals" if dim == "keypoints" else "keypoints"
    elem1 = getattr(a, dim).item()
    elem2 = getattr(b, dim).item()
    a = _validate_core_dimension(a, core_dim)
    b = _validate_core_dimension(b, core_dim)
    result = xr.apply_ufunc(
        cdist,
        a,
        b,
        kwargs={"metric": metric, **kwargs},
        input_core_dims=[[core_dim, "space"], [core_dim, "space"]],
        output_core_dims=[[elem1, elem2]],
        vectorize=True,
    )
    result = result.assign_coords(
        {
            elem1: getattr(a, core_dim).values,
            elem2: getattr(a, core_dim).values,
        }
    )
    # Drop any squeezed coordinates
    return result.squeeze(drop=True)


def compute_pairwise_distances(
    data: xr.DataArray,
    dim: Literal["individuals", "keypoints"],
    pairs: dict[str, str | list[str]] | Literal["all"],
    metric: str | None = "euclidean",
    **kwargs,
) -> xr.DataArray | dict[str, xr.DataArray]:
    """Compute pairwise distances between ``individuals`` or ``keypoints``.

    This function computes the distances between
    pairs of ``individuals`` (i.e. interindividual distances) or
    pairs of ``keypoints`` (i.e. interkeypoint distances),
    as determined by ``dim``.
    The distances are computed for the given ``pairs``
    using the specified ``metric``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing
        ``('time', 'individuals', 'keypoints', 'space')`` as dimensions.
    dim : Literal["individuals", "keypoints"]
        The dimension to compute the distances for. Must be either
        ``'individuals'`` or ``'keypoints'``.
    pairs : dict[str, str | list[str]] | Literal["all"]
        A dictionary containing the mapping between pairs of ``individuals``
        or ``keypoints``, or the special keyword ``"all"``.

        - If a dictionary is provided,
          the key is the keypoint or individual and
          the value can be a list of keypoints or individuals,
          or a string representing a single keypoint or individual
          to compute the distance with.
        - If the special keyword ``'all'`` is provided,
          all possible combinations of pairs are computed.
    metric : str, optional
        The distance metric to use. Must be one of the options supported
        by :func:`scipy.spatial.distance.cdist`, e.g. ``'cityblock'``,
        ``'euclidean'``, etc.
        Defaults to ``'euclidean'``.
    **kwargs : dict
        Additional keyword arguments to pass to
        :func:`scipy.spatial.distance.cdist`.

    Returns
    -------
    xarray.DataArray | dict[str, xarray.DataArray]
        An :class:`xarray.DataArray` containing the computed distances between
        a single pair of ``individuals`` or ``keypoints``. If multiple
        ``pairs`` are provided, this will be a dictionary containing the
        computed distances for each pair, with the key being the pair of
        keypoints or individuals and the value being the
        :class:`xarray.DataArray` containing the computed distances.

    Raises
    ------
    ValueError
        If ``dim`` is not one of ``'individuals'`` or ``'keypoints'``;
        if ``pairs`` is not a dictionary or ``'all'``; or
        if there are no pairs in ``data`` to compute distances for.

    Examples
    --------
    Compute the Euclidean distance (default) for all keypoints
    between ``ind1`` and ``ind2`` (i.e. interindividual distance):

    >>> position = xr.DataArray(
    ...     np.arange(36).reshape(2, 3, 3, 2),
    ...     coords={
    ...         "time": np.arange(2),
    ...         "individuals": ["ind1", "ind2", "ind3"],
    ...         "keypoints": ["key1", "key2", "key3"],
    ...         "space": ["x", "y"],
    ...     },
    ...     dims=["time", "individuals", "keypoints", "space"],
    ... )
    >>> dist_ind1_ind2 = compute_pairwise_distances(
    ...     position, "individuals", {"ind1": "ind2"}
    ... )
    >>> dist_ind1_ind2
    <xarray.DataArray (time: 2, ind1: 3, ind2: 3)> Size: 144B
    8.485 11.31 14.14 5.657 8.485 11.31 ... 5.657 8.485 11.31 2.828 5.657 8.485
    Coordinates:
    * time     (time) int64 16B 0 1
    * ind1     (ind1) <U4 48B 'key1' 'key2' 'key3'
    * ind2     (ind2) <U4 48B 'key1' 'key2' 'key3'

    The resulting ``dist_ind1_ind2`` is a DataArray containing the computed
    distances between ``ind1`` and ``ind2`` for all keypoints
    at each time point.

    To obtain the distances between ``key1`` of ``ind1`` and
    ``key2`` of ``ind2``:

    >>> dist_ind1_ind2.sel(ind1="key1", ind2="key2")

    Compute the Euclidean distance (default) between ``key1`` and ``key2``
    for all pairs of individuals and within each individual
    (i.e. interkeypoint distance):

    >>> dist_key1_key2 = compute_pairwise_distances(
    ...     position, "keypoints", {"key1": "key2"}
    ... )
    >>> dist_key1_key2
    <xarray.DataArray (time: 2, key1: 3, key2: 3)> Size: 144B
    2.828 11.31 19.8 5.657 2.828 11.31 14.14 ... 2.828 11.31 14.14 5.657 2.828
    Coordinates:
    * time     (time) int64 16B 0 1
    * key1     (key1) <U4 48B 'ind1' 'ind2' 'ind3'
    * key2     (key2) <U4 48B 'ind1' 'ind2' 'ind3'

    The resulting ``dist_key1_key2`` is a DataArray containing the computed
    distances between ``key1`` and ``key2`` for all individuals
    at each time point.

    To obtain the distances between ``key1`` and ``key2`` within ``ind1``:

    >>> dist_key1_key2.sel(key1="ind1", key2="ind1")

    To obtain the distances between ``key1`` of ``ind1`` and
    ``key2`` of ``ind2``:

    >>> dist_key1_key2.sel(key1="ind1", key2="ind2")

    Compute the city block or Manhattan distance for multiple pairs of
    keypoints using ``position``:

    >>> key_dists = compute_pairwise_distances(
    ...     position,
    ...     "keypoints",
    ...     {"key1": "key2", "key3": ["key1", "key2"]},
    ...     metric="cityblock",
    ... )
    >>> key_dists.keys()
    dict_keys(['dist_key1_key2', 'dist_key3_key1', 'dist_key3_key2'])

    As multiple pairs of keypoints are specified,
    the resulting ``key_dists`` is a dictionary containing the DataArrays
    of computed distances for each pair of keypoints.

    Compute the city block or Manhattan distance for all possible pairs of
    individuals using ``position``:

    >>> ind_dists = compute_pairwise_distances(
    ...     position,
    ...     "individuals",
    ...     "all",
    ...     metric="cityblock",
    ... )
    >>> ind_dists.keys()
    dict_keys(['dist_ind1_ind2', 'dist_ind1_ind3', 'dist_ind2_ind3'])

    See Also
    --------
    scipy.spatial.distance.cdist : The underlying function used.

    """
    if dim not in ["individuals", "keypoints"]:
        raise log_error(
            ValueError,
            "'dim' must be either 'individuals' or 'keypoints', "
            f"but got {dim}.",
        )
    if isinstance(pairs, str) and pairs != "all":
        raise log_error(
            ValueError,
            f"'pairs' must be a dictionary or 'all', but got {pairs}.",
        )
    # Find all possible pair combinations if "all" is specified
    if pairs == "all":
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
    if not paired_elements:
        raise log_error(
            ValueError, "Could not find any pairs to compute distances for."
        )
    pairwise_distances = {
        f"dist_{elem1}_{elem2}": _cdist(
            data.sel({dim: elem1}),
            data.sel({dim: elem2}),
            dim=dim,
            metric=metric,
            **kwargs,
        )
        for elem1, elem2 in paired_elements
    }
    # Return DataArray if result only has one key
    if len(pairwise_distances) == 1:
        return next(iter(pairwise_distances.values()))
    return pairwise_distances


def _validate_core_dimension(
    data: xr.DataArray, core_dim: str
) -> xr.DataArray:
    """Validate the input data contains the required core dimension.

    This function ensures the input data contains the ``core_dim``
    required when applying :func:`scipy.spatial.distance.cdist` to
    the input data, by adding a temporary dimension if necessary.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to validate.
    core_dim : str
        The core dimension to validate.

    Returns
    -------
    xarray.DataArray
        The input data with the core dimension validated.

    """
    if data.coords.get(core_dim) is None:
        data = data.assign_coords({core_dim: "temp_dim"})
    if data.coords[core_dim].ndim == 0:
        data = data.expand_dims(core_dim).transpose("time", "space", core_dim)
    return data
