"""Compute kinematic variables like velocity and acceleration."""

import itertools
from typing import Literal

import xarray as xr
from scipy.spatial.distance import cdist as _cdist

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


def cdist(
    a: xr.DataArray,
    b: xr.DataArray,
    dim: Literal["individuals", "keypoints"],
    metric: str | None = "euclidean",
    **kwargs,
) -> xr.DataArray:
    """Compute distance between each pair of the two collections of inputs.

    This function is a wrapper around :func:`scipy.spatial.distance.cdist`
    and computes the pairwise distances between either a pair of
    ``individuals`` or ``keypoints`` as specified by ``dim``.
    The distances are computed using the specified ``metric``.

    Parameters
    ----------
    a : xarray.DataArray
        The first input data containing position vectors of a
        single individual or keypoint, with ``space`` as a dimension.
    b : xarray.DataArray
        The second input data containing position vectors of a
        single individual or keypoint, with ``space`` as a dimension.
    dim : str
        The dimension to compute the distances for. Must be either
        ``'individuals'`` or ``'keypoints'``.
    metric : str, optional
        The distance metric to use. Must be one of the options supported
        by :func:`scipy.spatial.distance.cdist`, i.e.
        ``'braycurtis'``, ``'canberra'``, ``'chebyshev'``, ``'cityblock'``,
        ``'correlation'``, ``'cosine'``, ``'dice'``, ``'euclidean'``,
        ``'hamming'``, ``'jaccard'``, ``'jensenshannon'``, ``'kulczynski1'``,
        ``'mahalanobis'``, ``'matching'``, ``'minkowski'``,
        ``'rogerstanimoto'``, ``'russellrao'``, ``'seuclidean'``,
        ``'sokalmichener'``, ``'sokalsneath'``, ``'sqeuclidean'``, ``'yule'``.
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
    >>> ind_dists = cdist(pos1, pos2, dim="individuals")

    Compute the Euclidean distance (default) between ``key1`` and
    ``key2`` (i.e. interkeypoint distance for all individuals)
    using the ``position`` data variable in the Dataset ``ds``:

    >>> pos1 = ds.position.sel(keypoints="key1")
    >>> pos2 = ds.position.sel(keypoints="key2")
    >>> key_dists = cdist(pos1, pos2, dim="keypoints")

    Obtain the distance between ``key1`` of ``ind1`` and
    ``key2`` of ``ind2`` from ``ind_dists``
    (i.e. interindividual distance, different keypoints):

    >>> dist_ind1key1_ind2key2 = ind_dists.sel(ind1="key1", ind2="key2")

    Equivalently, the same distance can be obtained from ``key_dists``:

    >>> dist_ind1key1_ind2key2 = key_dists.sel(key1="ind1", key2="ind2")

    Obtain the distance between ``key1`` and ``key2`` of ``ind1``
    (i.e. interkeypoint distance within the same individual):

    >>> dist_ind1key1_ind1key2 = key_dists.sel(key1="ind1", key2="ind1")

    Obtain the distance between ``key1`` of ``ind1`` and ``ind2``
    (i.e. interindividual distance, same keypoint):

    >>> dist_ind1key1_ind2key1 = ind_dists.sel(ind1="key1", ind2="key1")

    See Also
    --------
    scipy.spatial.distance.cdist : The underlying function used.
    compute_interindividual_distances : Compute distances between one or
        more pairs of individuals within and across all keypoints.
    compute_interkeypoint_distances : Compute distances between one or
        more pairs of keypoints within and across all individuals.

    """
    core_dim = "individuals" if dim == "keypoints" else "keypoints"
    elem1 = getattr(a, dim).item()
    elem2 = getattr(b, dim).item()
    a = _validate_core_dimension(a, core_dim)
    b = _validate_core_dimension(b, core_dim)
    result = xr.apply_ufunc(
        _cdist,
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


def compute_interindividual_distances(
    data: xr.DataArray,
    pairs: dict[str, str | list[str]] | None = None,
    metric: str | None = "euclidean",
) -> xr.DataArray | dict[str, xr.DataArray]:
    """Compute interindividual distances for all pairs of keypoints.

    Distances are computed between pairs of individuals for all possible
    combinations of keypoint pairs at each time point.
    The distances are computed using the specified ``metric``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing
        ``('time', 'individuals', 'keypoints', 'space')`` as dimensions.
    pairs : dict[str, str | list[str]], optional
        A dictionary containing the mapping between pairs of individuals.
        The key is the individual and the value is a list of individuals
        or a string representing an individual to compute the distance with.
        If not provided, defaults to ``None`` and all possible combinations
        of pairs are computed.
    metric : str, optional
        The distance metric to use. Must be one of the options supported
        by :func:`scipy.spatial.distance.cdist`, i.e.
        ``'braycurtis'``, ``'canberra'``, ``'chebyshev'``, ``'cityblock'``,
        ``'correlation'``, ``'cosine'``, ``'dice'``, ``'euclidean'``,
        ``'hamming'``, ``'jaccard'``, ``'jensenshannon'``, ``'kulczynski1'``,
        ``'mahalanobis'``, ``'matching'``, ``'minkowski'``,
        ``'rogerstanimoto'``, ``'russellrao'``, ``'seuclidean'``,
        ``'sokalmichener'``, ``'sokalsneath'``, ``'sqeuclidean'``, ``'yule'``.
        Defaults to ``'euclidean'``.

    Returns
    -------
    xarray.DataArray | dict[str, xarray.DataArray]
        An :class:`xarray.DataArray` containing the computed distances for
        all keypoints between the given pair of individuals, or if
        multiple pairs are provided, a dictionary containing the computed
        distances for each pair of individuals, with the key being the pair
        of individuals and the value being the :class:`xarray.DataArray`
        containing the computed distances.

    Examples
    --------
    Compute the Euclidean distance (default) for all keypoints
    between ``ind1`` and ``ind2`` (i.e. interindividual distance):

    >>> position = xr.DataArray(
    ...     np.arange(24).reshape(2, 3, 2, 2),
    ...     coords={
    ...         "time": np.arange(2),
    ...         "individuals": ["ind1", "ind2", "ind3"],
    ...         "keypoints": ["key1", "key2"],
    ...         "space": ["x", "y"],
    ...     },
    ...     dims=["time", "individuals", "keypoints", "space"],
    ... )
    >>> dist_ind1_ind2 = compute_interindividual_distances(
    ...     position, pairs={"ind1": "ind2"}
    ... )
    >>> dist_ind1_ind2
    <xarray.DataArray (time: 2, ind1: 2, ind2: 2)> Size: 64B
    5.657 8.485 2.828 5.657 5.657 8.485 2.828 5.657
    Coordinates:
      * time     (time) int32 8B 0 1
      * ind1     (ind1) <U4 32B 'key1' 'key2'
      * ind2     (ind2) <U4 32B 'key1' 'key2'

    The resulting ``dist_ind1_ind2`` is a DataArray containing the computed
    distances between ``ind1`` and ``ind2`` for all keypoints
    at each time point.

    To obtain the distances between ``key1`` of ``ind1`` and
    ``key2`` of ``ind2``:

    >>> dist_ind1_ind2.sel(ind1="key1", ind2="key2")

    Compute the city block or Manhattan distance for multiple pairs of
    individuals using ``position``:

    >>> ind_dists = compute_interindividual_distances(
    ...     position,
    ...     pairs={"ind1": "ind2", "ind3": ["ind1", "ind2"]},
    ...     metric="cityblock",
    ... )
    >>> ind_dists.keys()
    dict_keys(['dist_ind1_ind2', 'dist_ind3_ind1', 'dist_ind3_ind2'])

    As multiple pairs of individuals are specified,
    the resulting ``ind_dists`` is a dictionary containing the DataArrays
    of computed distances for each pair of individuals.

    Compute the city block or Manhattan distance for all possible pairs of
    individuals using ``position``:

    >>> ind_dists = compute_interindividual_distances(
    ...     position,
    ...     metric="cityblock",
    ... )
    >>> ind_dists.keys()
    dict_keys(['dist_ind1_ind2', 'dist_ind1_ind3', 'dist_ind2_ind3'])

    See Also
    --------
    compute_interkeypoint_distances : Compute distances between one or
        more pairs of keypoints within and across all individuals.

    """
    return _compute_pairwise_distances(
        data, "individuals", pairs=pairs, metric=metric
    )


def compute_interkeypoint_distances(
    data: xr.DataArray,
    pairs: dict[str, str | list[str]] | None = None,
    metric: str | None = "euclidean",
) -> xr.DataArray | dict[str, xr.DataArray]:
    """Compute interkeypoint distances within and across all individuals.

    Distances are computed between pairs of keypoints for all possible
    combinations of individual pairs at each time point.
    The distances are computed using the specified ``metric``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing
        ``('time', 'individuals', 'keypoints', 'space')`` as dimensions.
    pairs : dict[str, str | list[str]], optional
        A dictionary containing the mapping between pairs of keypoints.
        The key is the keypoint and the value is a list of keypoints
        or a string representing a keypoint to compute the distance with.
        If not provided, defaults to ``None`` and all possible combinations
        of pairs are computed.
    metric : str, optional
        The distance metric to use. Must be one of the options supported
        by :func:`scipy.spatial.distance.cdist`, i.e.
        ``'braycurtis'``, ``'canberra'``, ``'chebyshev'``, ``'cityblock'``,
        ``'correlation'``, ``'cosine'``, ``'dice'``, ``'euclidean'``,
        ``'hamming'``, ``'jaccard'``, ``'jensenshannon'``, ``'kulczynski1'``,
        ``'mahalanobis'``, ``'matching'``, ``'minkowski'``,
        ``'rogerstanimoto'``, ``'russellrao'``, ``'seuclidean'``,
        ``'sokalmichener'``, ``'sokalsneath'``, ``'sqeuclidean'``, ``'yule'``.
        Defaults to ``'euclidean'``.

    Returns
    -------
    xarray.DataArray | dict[str, xarray.DataArray]
        An :class:`xarray.DataArray` containing the computed distances for
        all individuals between the given pair of keypoints, or if
        multiple pairs are provided, a dictionary containing the computed
        distances for each pair of keypoints, with the key being the pair
        of keypoints and the value being the :class:`xarray.DataArray`
        containing the computed distances.

    Examples
    --------
    Compute the Euclidean distance (default) for all keypoints
    between ``ind1`` and ``ind2`` (i.e. interindividual distance):

    >>> position = xr.DataArray(
    ...     np.arange(24).reshape(2, 2, 3, 2),
    ...     coords={
    ...         "time": np.arange(2),
    ...         "individuals": ["ind1", "ind2"],
    ...         "keypoints": ["key1", "key2", "key3"],
    ...         "space": ["x", "y"],
    ...     },
    ...     dims=["time", "individuals", "keypoints", "space"],
    ... )
    >>> dist_key1_key2 = compute_interkeypoint_distances(
    ...     position, pairs={"key1": "key2"}
    ... )
    >>> dist_key1_key2
    <xarray.DataArray (time: 2, key1: 2, key2: 2)> Size: 64B
    2.828 11.31 5.657 2.828 2.828 11.31 5.657 2.828
    Coordinates:
      * time     (time) int32 8B 0 1
      * key1     (key1) <U4 32B 'ind1' 'ind2'
      * key2     (key2) <U4 32B 'ind1' 'ind2'

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

    >>> key_dists = compute_interkeypoint_distances(
    ...     position,
    ...     pairs={"key1": "key2", "key3": ["key1", "key2"]},
    ...     metric="cityblock",
    ... )
    >>> key_dists.keys()
    dict_keys(['dist_key1_key2', 'dist_key3_key1', 'dist_key3_key2'])

    As multiple pairs of keypoints are specified,
    the resulting ``key_dists`` is a dictionary containing the DataArrays
    of computed distances for each pair of keypoints.

    Compute the city block or Manhattan distance for all possible pairs of
    keypoints using ``position``:

    >>> key_dists = compute_interkeypoint_distances(
    ...     position,
    ...     metric="cityblock",
    ... )
    >>> key_dists.keys()
    dict_keys(['dist_key1_key2', 'dist_key1_key3', 'dist_key2_key3'])

    See Also
    --------
    compute_interindividual_distances : Compute distances between one or
        more pairs of individuals within and across all keypoints.

    """
    return _compute_pairwise_distances(
        data, "keypoints", pairs=pairs, metric=metric
    )


def _compute_pairwise_distances(
    data: xr.DataArray,
    dim: Literal["individuals", "keypoints"],
    pairs: dict[str, str | list[str]] | None = None,
    metric: str | None = "euclidean",
    **kwargs,
) -> xr.DataArray | dict[str, xr.DataArray]:
    """Compute pairwise distances between ``individuals`` or ``keypoints``.

    This function computes the distances between pairs of ``keypoints``
    (i.e. interkeypoint distances) or pairs of ``individuals`` (i.e.
    interindividual distances). The distances are computed using the
    specified ``metric``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing
        ``('time', 'individuals', 'keypoints', 'space')`` as dimensions.
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
    metric : str, optional
        The distance metric to use. Must be one of the options supported
        by :func:`scipy.spatial.distance.cdist`, i.e.
        ``'braycurtis'``, ``'canberra'``, ``'chebyshev'``, ``'cityblock'``,
        ``'correlation'``, ``'cosine'``, ``'dice'``, ``'euclidean'``,
        ``'hamming'``, ``'jaccard'``, ``'jensenshannon'``, ``'kulczynski1'``,
        ``'mahalanobis'``, ``'matching'``, ``'minkowski'``,
        ``'rogerstanimoto'``, ``'russellrao'``, ``'seuclidean'``,
        ``'sokalmichener'``, ``'sokalsneath'``, ``'sqeuclidean'``, ``'yule'``.
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

    See Also
    --------
    cdist : ``movement``'s wrapper around :func:`scipy.spatial.distance.cdist`.
    :func:`scipy.spatial.distance.cdist` : The underlying function used.

    """
    if dim not in ["individuals", "keypoints"]:
        raise log_error(
            ValueError,
            "'dim' must be either 'individuals' or 'keypoints', "
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
        input1 = data.sel({dim: elem1})
        input2 = data.sel({dim: elem2})
        pairwise_distances[f"dist_{elem1}_{elem2}"] = cdist(
            input1, input2, dim=dim, metric=metric, **kwargs
        )
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
