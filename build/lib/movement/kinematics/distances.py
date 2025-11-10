"""Computing spatial relationships between points, such as distances."""

import itertools
from typing import Literal

import xarray as xr
from scipy.spatial.distance import cdist

from movement.utils.logging import logger
from movement.validators.arrays import validate_dims_coords


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
        single individual or keypoint, with ``time``, ``space``
        (in Cartesian coordinates), and ``individuals`` or ``keypoints``
        (as specified by ``dim``) as required dimensions.
    b : xarray.DataArray
        The second input data containing position information of a
        single individual or keypoint, with ``time``, ``space``
        (in Cartesian coordinates), and ``individuals`` or ``keypoints``
        (as specified by ``dim``) as required dimensions.
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
    # The dimension from which ``dim`` labels are obtained
    labels_dim = "individuals" if dim == "keypoints" else "keypoints"
    elem1 = getattr(a, dim).item()
    elem2 = getattr(b, dim).item()
    a = _validate_labels_dimension(a, labels_dim)
    b = _validate_labels_dimension(b, labels_dim)
    result = xr.apply_ufunc(
        cdist,
        a,
        b,
        kwargs={"metric": metric, **kwargs},
        input_core_dims=[[labels_dim, "space"], [labels_dim, "space"]],
        output_core_dims=[[elem1, elem2]],
        vectorize=True,
    )
    result = result.assign_coords(
        {
            elem1: getattr(a, labels_dim).values,
            elem2: getattr(a, labels_dim).values,
        }
    )
    result.name = "distance"
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
        The input data containing position information, with ``time``,
        ``space`` (in Cartesian coordinates), and
        ``individuals`` or ``keypoints`` (as specified by ``dim``)
        as required dimensions.
    dim : Literal["individuals", "keypoints"]
        The dimension to compute the distances for. Must be either
        ``'individuals'`` or ``'keypoints'``.
    pairs : dict[str, str | list[str]] or 'all'
        Specifies the pairs of elements (either individuals or keypoints)
        for which to compute distances, depending on the value of ``dim``.

        - If ``dim='individuals'``, ``pairs`` should be a dictionary where
          each key is an individual name, and each value is also an individual
          name or a list of such names to compute distances with.
        - If ``dim='keypoints'``, ``pairs`` should be a dictionary where each
          key is a keypoint name, and each value is also keypoint name or a
          list of such names to compute distances with.
        - Alternatively, use the special keyword ``'all'`` to compute distances
          for all possible pairs of individuals or keypoints
          (depending on ``dim``).
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
    xarray.DataArray or dict[str, xarray.DataArray]
        The computed pairwise distances. If a single pair is specified in
        ``pairs``, returns an :class:`xarray.DataArray`. If multiple pairs
        are specified, returns a dictionary where each key is a string
        representing the pair  (e.g., ``'dist_ind1_ind2'`` or
        ``'dist_key1_key2'``) and each value is an :class:`xarray.DataArray`
        containing the computed distances for that pair.

    Raises
    ------
    ValueError
        If ``dim`` is not one of ``'individuals'`` or ``'keypoints'``;
        if ``pairs`` is not a dictionary or ``'all'``; or
        if there are no pairs in ``data`` to compute distances for.

    Examples
    --------
    Compute the Euclidean distance (default) between ``ind1`` and ``ind2``
    (i.e. interindividual distance), for all possible pairs of keypoints.

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
    (i.e. interkeypoint distance), for all possible pairs of individuals.

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
        raise logger.error(
            ValueError(
                "'dim' must be either 'individuals' or 'keypoints', "
                f"but got {dim}."
            )
        )
    if isinstance(pairs, str) and pairs != "all":
        raise logger.error(
            ValueError(
                f"'pairs' must be a dictionary or 'all', but got {pairs}."
            )
        )
    validate_dims_coords(data, {"time": [], "space": ["x", "y"], dim: []})
    # Find all possible pair combinations if 'all' is specified
    if pairs == "all":
        paired_elements = list(
            itertools.combinations(getattr(data, dim).values, 2)
        )
    else:
        paired_elements = [
            (elem1, elem2)
            for elem1, elem2_list in pairs.items()
            for elem2 in (
                # Ensure elem2_list is a list
                [elem2_list] if isinstance(elem2_list, str) else elem2_list
            )
        ]
    if not paired_elements:
        raise logger.error(
            ValueError("Could not find any pairs to compute distances for.")
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


def _validate_labels_dimension(data: xr.DataArray, dim: str) -> xr.DataArray:
    """Validate the input data contains the ``dim`` for labelling dimensions.

    This function ensures the input data contains the ``dim``
    used as labels (coordinates) when applying
    :func:`scipy.spatial.distance.cdist` to
    the input data, by adding a temporary dimension if necessary.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to validate.
    dim : str
        The dimension to validate.

    Returns
    -------
    xarray.DataArray
        The input data with the labels dimension validated.

    """
    if data.coords.get(dim) is None:
        data = data.assign_coords({dim: "temp_dim"})
    if data.coords[dim].ndim == 0:
        data = data.expand_dims(dim).transpose("time", "space", dim)
    return data
