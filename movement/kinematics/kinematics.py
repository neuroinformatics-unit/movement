"""Compute core kinematic variables such as time derivatives of ``position``.

This module provides functions for computing fundamental kinematic properties
such as forward & backward displacement, velocity, acceleration, speed, and
path length (distance travelled between two time points).
The ``movement.kinematics`` subpackage encompasses a broader range of
functionality (e.g., orientations and distances ), but this file is
intended to isolate 'true' kinematics for clarity. In a future release, the
public API may be revised to reflect this distinction more explicitly.

"""

import warnings
from typing import Literal

import numpy as np
import xarray as xr

from movement.utils.logging import logger
from movement.utils.reports import report_nan_values
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


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
    data : xarray.DataArray
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
    result = result.reindex(data.coords, fill_value=0)
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
    result = result.reindex(data.coords, fill_value=0)
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
    data : xarray.DataArray
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
    data : xarray.DataArray
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
    data : xarray.DataArray
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
    data : xarray.DataArray
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
    result = compute_norm(compute_velocity(data))
    result.name = "speed"
    return result


def compute_path_length(
    data: xr.DataArray,
    start: float | None = None,
    stop: float | None = None,
    nan_policy: Literal["ffill", "scale"] = "ffill",
    nan_warn_threshold: float = 0.2,
) -> xr.DataArray:
    r"""Compute the length of a path travelled between two time points.

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
        If any point track in the data has at least (:math:`\ge`)
        this proportion of values missing, a warning will be emitted.
        Defaults to 0.2 (20%).

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
    n_time = data.sizes["time"]
    if n_time < 2:
        raise logger.error(
            ValueError(
                "At least 2 time points are required to compute path length, "
                f"but {n_time} were found. "
                "Double-check the start and stop times."
            )
        )

    _warn_about_nan_proportion(data, nan_warn_threshold)

    if nan_policy == "ffill":
        result = compute_norm(
            compute_backward_displacement(data.ffill(dim="time")).isel(
                time=slice(1, None)
            )
        ).sum(dim="time", min_count=1)
    elif nan_policy == "scale":
        result = _compute_scaled_path_length(data)
    else:
        raise logger.error(
            ValueError(
                f"Invalid value for nan_policy: {nan_policy}. "
                "Must be one of 'ffill' or 'scale'."
            )
        )

    result.name = "path_length"
    return result


def _warn_about_nan_proportion(
    data: xr.DataArray, nan_warn_threshold: float
) -> None:
    """Issue warning if the proportion of NaN values exceeds a threshold.

    The NaN proportion is evaluated per point track, and a given point is
    considered NaN if any of its ``space`` coordinates are NaN. The warning
    specifically lists the point tracks with at least (>=)
    ``nan_warn_threshold`` proportion of NaN values.

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
        raise logger.error(
            ValueError("nan_warn_threshold must be between 0 and 1.")
        )
    n_nans = data.isnull().any(dim="space").sum(dim="time")
    data_to_warn_about = data.where(
        n_nans >= data.sizes["time"] * nan_warn_threshold, drop=True
    )
    if data_to_warn_about.size > 0:
        warnings.warn(
            "The result may be unreliable for point tracks with many "
            "missing values. The following tracks have at least "
            f"{nan_warn_threshold * 100:.3} % NaN values:\n"
            f"{report_nan_values(data_to_warn_about)}",
            UserWarning,
            stacklevel=2,
        )


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
    displacement = compute_backward_displacement(data).isel(
        time=slice(1, None)
    )
    valid_segments = (~displacement.isnull()).all(dim="space").sum(dim="time")
    valid_proportion = valid_segments / (data.sizes["time"] - 1)
    return compute_norm(displacement).sum(dim="time") / valid_proportion


def _make_nan_result(name: str) -> xr.DataArray:
    result = xr.DataArray(np.nan, attrs={"units": "dimensionless"})
    result.name = name
    return result


def compute_sinuosity(
    data: xr.DataArray,
    nan_policy: Literal["ffill", "scale"] = "scale",
    nan_warn_threshold: float = 0.2,
) -> xr.DataArray:
    """Compute Benhamou's sinuosity index for trajectory data.

    Sinuosity measures the tortuosity of a path as the ratio of the
    path length to the net displacement, with values closer to 1
    indicating straighter paths.

    Parameters
    ----------
    data : xr.DataArray
        The input data containing position information, with
        ``time`` and ``space`` (containing ``x`` and ``y``)
        as required dimensions.
    nan_policy : Literal["ffill", "scale"], optional
        Policy to handle NaN values. Default is "scale".
    nan_warn_threshold : float, optional
        Proportion of NaN values above which to issue a warning.
        Default is 0.2 (20%).

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the sinuosity index.

    References
    ----------
    Benhamou, S. (2004). How to reliably estimate the tortuosity of
    an animal's path. Journal of Theoretical Biology, 229(2), 209-220.

    """
    validate_dims_coords(data, {"time": [], "space": []})

    if _is_stationary_trajectory(data):
        result = xr.DataArray(
            np.nan,
            attrs={"units": "dimensionless"},
        )
        result.name = "sinuosity"
        return result

    if _is_stationary_trajectory(data) or not _filter_valid_regions(data):
        return _make_nan_result("sinuosity")

    _warn_about_nan_proportion(data, nan_warn_threshold)

    path_length = compute_path_length(data, nan_policy=nan_policy)

    net_displacement = _compute_net_displacement(data)

    sinuosity = xr.where(
        net_displacement > 0,
        path_length / net_displacement,
        np.nan,
    )

    sinuosity.name = "sinuosity"
    return sinuosity


def _compute_net_displacement(data: xr.DataArray) -> xr.DataArray:
    """Compute net displacement (start-to-end Euclidean distance)."""
    start_pos = _get_first_valid_position(data)
    end_pos = _get_last_valid_position(data)
    diff = end_pos - start_pos
    dist = np.sqrt((diff**2).sum(dim="space"))
    return xr.DataArray(dist)


def _get_first_valid_position(data: xr.DataArray) -> xr.DataArray:
    """Get the first valid (non-NaN) position for each trajectory.

    Parameters
    ----------
    data : xr.DataArray
        Position data with time and space dimensions.

    Returns
    -------
    xr.DataArray
        First valid position for each individual/keypoint.

    """
    valid_mask = ~data.isnull().any(dim="space")
    first_valid_idx = valid_mask.argmax(dim="time")

    return data.isel(time=first_valid_idx)


def _get_last_valid_position(data: xr.DataArray) -> xr.DataArray:
    """Get the last valid (non-NaN) position for each trajectory.

    Parameters
    ----------
    data : xarray.DataArray
        Position data with time and space dimensions.

    Returns
    -------
    xarray.DataArray
        Last valid position for each individual/keypoint.

    """
    valid_mask = ~data.isnull().any(dim="space")

    return data.where(valid_mask, drop=True).isel(time=-1)


def compute_straightness_index(
    data: xr.DataArray,
    nan_policy: Literal["ffill", "scale"] = "scale",
    nan_warn_threshold: float = 0.2,
) -> xr.DataArray:
    """Compute straightness index for trajectory data.

    The straightness index is the inverse of sinuosity, ranging from
    0 (highly tortuous) to 1 (perfectly straight).

    Parameters
    ----------
    data : xr.DataArray
        The input data containing position information, with
        ``time`` and ``space`` as required dimensions.
    nan_policy : Literal["ffill", "scale"], optional
        Policy to handle NaN values. Default is "scale".
    nan_warn_threshold : float, optional
        Proportion of NaN values above which to issue a warning.
        Default is 0.2 (20%).

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the straightness index.

    Notes
    -----
    Straightness Index = net_displacement / path_length
    This is the reciprocal of Benhamou's sinuosity.

    """
    validate_dims_coords(data, {"time": [], "space": []})

    if _is_stationary_trajectory(data):
        result = xr.DataArray(
            np.nan,
            attrs={"units": "dimensionless"},
        )
        result.name = "straightness_index"
        return result

    if _is_stationary_trajectory(data) or not _filter_valid_regions(data):
        return _make_nan_result("straightness_index")

    _warn_about_nan_proportion(data, nan_warn_threshold)

    path_length = compute_path_length(data, nan_policy=nan_policy)
    net_displacement = _compute_net_displacement(data)
    straightness = xr.where(
        path_length > 0,
        net_displacement / path_length,
        np.nan,
    )
    straightness.name = "straightness_index"
    return straightness


def _filter_valid_regions(
    data: xr.DataArray,
    min_valid_points: int = 2,
) -> bool:
    """Check if trajectory has sufficient valid (non-NaN) data points.

    Parameters
    ----------
    data : xr.DataArray
        The input data containing position information.
    min_valid_points : int, optional
        Minimum number of valid points required for a meaningful
        computation. Default is 2.

    Returns
    -------
    bool
        True if trajectory has sufficient valid data, False otherwise.

    """
    valid_mask = ~data.isnull().any(dim="space")
    valid_count = valid_mask.sum(dim="time")
    return bool(valid_count >= min_valid_points)


def _is_stationary_trajectory(
    data: xr.DataArray,
    tolerance: float = 1e-10,
) -> bool:
    """Check if a trajectory is stationary (no movement).

    Parameters
    ----------
    data : xr.DataArray
        The input data containing position information.
    tolerance : float, optional
        Tolerance for considering positions as identical.
        Default is 1e-10.

    Returns
    -------
    bool
        True if the trajectory is stationary, False otherwise.

    """
    valid_mask = ~data.isnull().any(dim="space")

    if not valid_mask.any():
        return True
    first_valid_idx = valid_mask.argmax(dim="time")
    first_pos = data.isel(time=first_valid_idx)
    valid_data = data.where(valid_mask, drop=True)
    if valid_data.sizes["time"] == 0:
        return True
    deviation = np.abs(valid_data - first_pos).max()
    return bool(deviation < tolerance)
