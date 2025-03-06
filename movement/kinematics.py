"""Compute kinematic variables like velocity and acceleration."""

import itertools
from collections.abc import Hashable
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist

from movement.utils.logging import log_error, log_warning
from movement.utils.reports import report_nan_values
from movement.utils.vector import (
    compute_norm,
    compute_signed_angle_2d,
    convert_to_unit,
)
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


def compute_speed(data: xr.DataArray) -> xr.DataArray:
    """Compute instantaneous speed at each time point.

    Speed is a scalar quantity computed as the Euclidean norm (magnitude)
    of the velocity vector at each time point.


    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed speed,
        with dimensions matching those of the input data,
        except ``space`` is removed.

    """
    return compute_norm(compute_velocity(data))


def compute_forward_vector(
    data: xr.DataArray,
    left_keypoint: Hashable,
    right_keypoint: Hashable,
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
) -> xr.DataArray:
    """Compute a 2D forward vector given two left-right symmetric keypoints.

    The forward vector is computed as a vector perpendicular to the
    line connecting two symmetrical keypoints on either side of the body
    (i.e., symmetrical relative to the mid-sagittal plane), and pointing
    forwards (in the rostral direction). A top-down or bottom-up view of the
    animal is assumed (see Notes).

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two symmetrical keypoints located on the left and
        right sides of the body, respectively.
    left_keypoint : Hashable
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint : Hashable
        Name of the right keypoint, e.g., "right_ear"
    camera_view : Literal["top_down", "bottom_up"], optional
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the forward vector, with
        dimensions matching the input data array, but without the
        ``keypoints`` dimension.

    Notes
    -----
    To determine the forward direction of the animal, we need to specify
    (1) the right-to-left direction of the animal and (2) its upward direction.
    We determine the right-to-left direction via the input left and right
    keypoints. The upwards direction, in turn, can be determined by passing the
    ``camera_view`` argument with either ``"top_down"`` or ``"bottom_up"``. If
    the camera view is specified as being ``"top_down"``, or if no additional
    information is provided, we assume that the upwards direction matches that
    of the vector ``[0, 0, -1]``. If the camera view is ``"bottom_up"``, the
    upwards direction is assumed to be given by ``[0, 0, 1]``. For both cases,
    we assume that position values are expressed in the image coordinate
    system (where the positive X-axis is oriented to the right, the positive
    Y-axis faces downwards, and positive Z-axis faces away from the person
    viewing the screen).

    If one of the required pieces of information is missing for a frame (e.g.,
    the left keypoint is not visible), then the computed head direction vector
    is set to NaN.

    """
    # Validate input data
    _validate_type_data_array(data)
    validate_dims_coords(
        data,
        {
            "time": [],
            "keypoints": [left_keypoint, right_keypoint],
            "space": [],
        },
    )
    if len(data.space) != 2:
        raise log_error(
            ValueError,
            "Input data must have exactly 2 spatial dimensions, but "
            f"currently has {len(data.space)}.",
        )
    # Validate input keypoints
    if left_keypoint == right_keypoint:
        raise log_error(
            ValueError, "The left and right keypoints may not be identical."
        )
    # Define right-to-left vector
    right_to_left_vector = data.sel(
        keypoints=left_keypoint, drop=True
    ) - data.sel(keypoints=right_keypoint, drop=True)
    # Define upward vector
    # default: negative z direction in the image coordinate system
    upward_vector = (
        np.array([0, 0, -1])
        if camera_view == "top_down"
        else np.array([0, 0, 1])
    )
    upward_vector = xr.DataArray(
        np.tile(upward_vector.reshape(1, -1), [len(data.time), 1]),
        dims=["time", "space"],
        coords={
            "space": ["x", "y", "z"],
        },
    )
    # Compute forward direction as the cross product
    # (right-to-left) cross (forward) = up
    forward_vector = xr.cross(
        right_to_left_vector, upward_vector, dim="space"
    ).drop_sel(
        space="z"
    )  # keep only the first 2 spatal dimensions of the result
    # Return unit vector
    return convert_to_unit(forward_vector)


def compute_head_direction_vector(
    data: xr.DataArray,
    left_keypoint: str,
    right_keypoint: str,
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
):
    """Compute the 2D head direction vector given two keypoints on the head.

    This function is an alias for :func:`compute_forward_vector()\
    <movement.kinematics.compute_forward_vector>`. For more
    detailed information on how the head direction vector is computed,
    please refer to the documentation for that function.

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
    camera_view : Literal["top_down", "bottom_up"], optional
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the head direction vector, with
        dimensions matching the input data array, but without the
        ``keypoints`` dimension.

    """
    return compute_forward_vector(
        data, left_keypoint, right_keypoint, camera_view=camera_view
    )


def compute_forward_vector_angle(
    data: xr.DataArray,
    left_keypoint: Hashable,
    right_keypoint: Hashable,
    reference_vector: xr.DataArray | ArrayLike = (1, 0),
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
    in_degrees: bool = False,
) -> xr.DataArray:
    r"""Compute the signed angle between a reference and a forward vector.

    Forward vector angle is the :func:`signed angle\
    <movement.utils.vector.compute_signed_angle_2d>`
    between the reference vector and the animal's :func:`forward vector\
    <movement.kinematics.compute_forward_vector>`.
    The returned angles are in radians, spanning the range :math:`(-\pi, \pi]`,
    unless ``in_degrees`` is set to ``True``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two symmetrical keypoints located on the left and
        right sides of the body, respectively.
    left_keypoint : Hashable
        Name of the left keypoint, e.g., "left_ear", used to compute the
        forward vector.
    right_keypoint : Hashable
        Name of the right keypoint, e.g., "right_ear", used to compute the
        forward vector.
    reference_vector : xr.DataArray | ArrayLike, optional
        The reference vector against which the ``forward_vector`` is
        compared to compute 2D heading. Must be a two-dimensional vector,
        in the form [x,y] - where ``reference_vector[0]`` corresponds to the
        x-coordinate and ``reference_vector[1]`` corresponds to the
        y-coordinate. If left unspecified, the vector [1, 0] is used by
        default.
    camera_view : Literal["top_down", "bottom_up"], optional
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.
    in_degrees : bool
        If ``True``, the returned heading array is given in degrees.
        Otherwise, the array is given in radians. Default ``False``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed forward vector angles,
        with dimensions matching the input data array,
        but without the ``keypoints`` and ``space`` dimensions.

    See Also
    --------
    movement.utils.vector.compute_signed_angle_2d :
        The underlying function used to compute the signed angle between two
        2D vectors. See this function for a definition of the signed
        angle between two vectors.
    movement.kinematics.compute_forward_vector :
        The function used to compute the forward vector.

    """
    # Convert reference vector to np.array if not already a valid array
    if not isinstance(reference_vector, np.ndarray | xr.DataArray):
        reference_vector = np.array(reference_vector)

    # Compute forward vector
    forward_vector = compute_forward_vector(
        data, left_keypoint, right_keypoint, camera_view=camera_view
    )

    # Compute signed angle between reference vector and forward vector
    heading_array = compute_signed_angle_2d(
        forward_vector, reference_vector, v_as_left_operand=True
    )

    # Convert to degrees
    if in_degrees:
        heading_array = np.rad2deg(heading_array)

    return heading_array


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


def compute_path_length(
    data: xr.DataArray,
    start: float | None = None,
    stop: float | None = None,
    nan_policy: Literal["ffill", "scale"] = "ffill",
    nan_warn_threshold: float = 0.2,
) -> xr.DataArray:
    """Compute the length of a path travelled between two time points.

    The path length is defined as the sum of the norms (magnitudes) of the
    displacement vectors between two time points ``start`` and ``stop``,
    which should be provided in the time units of the data array.
    If not specified, the minimum and maximum time coordinates of the data
    array are used as start and stop times, respectively.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    start : float, optional
        The start time of the path. If None (default),
        the minimum time coordinate in the data is used.
    stop : float, optional
        The end time of the path. If None (default),
        the maximum time coordinate in the data is used.
    nan_policy : Literal["ffill", "scale"], optional
        Policy to handle NaN (missing) values. Can be one of the ``"ffill"``
        or ``"scale"``. Defaults to ``"ffill"`` (forward fill).
        See Notes for more details on the two policies.
    nan_warn_threshold : float, optional
        If more than this proportion of values are missing in any point track,
        a warning will be emitted. Defaults to 0.2 (20%).

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed path length,
        with dimensions matching those of the input data,
        except ``time`` and ``space`` are removed.

    Notes
    -----
    Choosing ``nan_policy="ffill"`` will use :meth:`xarray.DataArray.ffill`
    to forward-fill missing segments (NaN values) across time.
    This equates to assuming that a track remains stationary for
    the duration of the missing segment and then instantaneously moves to
    the next valid position, following a straight line. This approach tends
    to underestimate the path length, and the error increases with the number
    of missing values.

    Choosing ``nan_policy="scale"`` will adjust the path length based on the
    the proportion of valid segments per point track. For example, if only
    80% of segments are present, the path length will be computed based on
    these and the result will be divided by 0.8. This approach assumes
    that motion dynamics are similar across observed and missing time
    segments, which may not accurately reflect actual conditions.

    """
    validate_dims_coords(data, {"time": [], "space": []})
    data = data.sel(time=slice(start, stop))
    # Check that the data is not empty or too short
    n_time = data.sizes["time"]
    if n_time < 2:
        raise log_error(
            ValueError,
            f"At least 2 time points are required to compute path length, "
            f"but {n_time} were found. Double-check the start and stop times.",
        )

    _warn_about_nan_proportion(data, nan_warn_threshold)

    if nan_policy == "ffill":
        return compute_norm(
            compute_displacement(data.ffill(dim="time")).isel(
                time=slice(1, None)
            )  # skip first displacement (always 0)
        ).sum(dim="time", min_count=1)  # return NaN if no valid segment
    elif nan_policy == "scale":
        return _compute_scaled_path_length(data)
    else:
        raise log_error(
            ValueError,
            f"Invalid value for nan_policy: {nan_policy}. "
            "Must be one of 'ffill' or 'scale'.",
        )


def _warn_about_nan_proportion(
    data: xr.DataArray, nan_warn_threshold: float
) -> None:
    """Print a warning if the proportion of NaN values exceeds a threshold.

    The NaN proportion is evaluated per point track, and a given point is
    considered NaN if any of its ``space`` coordinates are NaN. The warning
    specifically lists the point tracks that exceed the threshold.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array.
    nan_warn_threshold : float
        The threshold for the proportion of NaN values. Must be a number
        between 0 and 1.

    """
    nan_warn_threshold = float(nan_warn_threshold)
    if not 0 <= nan_warn_threshold <= 1:
        raise log_error(
            ValueError,
            "nan_warn_threshold must be between 0 and 1.",
        )
    n_nans = data.isnull().any(dim="space").sum(dim="time")
    data_to_warn_about = data.where(
        n_nans > data.sizes["time"] * nan_warn_threshold, drop=True
    )
    if len(data_to_warn_about) > 0:
        log_warning(
            "The result may be unreliable for point tracks with many "
            "missing values. The following tracks have more than "
            f"{nan_warn_threshold * 100:.3} % NaN values:",
        )
        print(report_nan_values(data_to_warn_about))


def _compute_scaled_path_length(
    data: xr.DataArray,
) -> xr.DataArray:
    """Compute scaled path length based on proportion of valid segments.

    Path length is first computed based on valid segments (non-NaN values
    on both ends of the segment) and then scaled based on the proportion of
    valid segments per point track - i.e. the result is divided by the
    proportion of valid segments.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed path length,
        with dimensions matching those of the input data,
        except ``time`` and ``space`` are removed.

    """
    # Skip first displacement segment (always 0) to not mess up the scaling
    displacement = compute_displacement(data).isel(time=slice(1, None))
    # count number of valid displacement segments per point track
    valid_segments = (~displacement.isnull()).all(dim="space").sum(dim="time")
    # compute proportion of valid segments per point track
    valid_proportion = valid_segments / (data.sizes["time"] - 1)
    # return scaled path length
    return compute_norm(displacement).sum(dim="time") / valid_proportion
