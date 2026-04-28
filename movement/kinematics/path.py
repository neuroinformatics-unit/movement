"""Compute path-level metrics such as path length and straightness.

By 'path' we refer to the spatial trajectory of an individual between two
time points. While these metrics can be computed based on any set of
keypoints, they are most meaningful when applied to a single keypoint
representing the individual's overall position (e.g., centroid).
"""

import warnings
from typing import Literal

import numpy as np
import xarray as xr

from movement.kinematics.kinematics import compute_backward_displacement
from movement.utils.logging import logger
from movement.utils.reports import report_nan_values
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


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
    # Check that the data is not empty or too short
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
            )  # skip first displacement (always 0)
        ).sum(dim="time", min_count=1)  # return NaN if no valid segment
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


def compute_path_straightness(
    data: xr.DataArray,
    start: float | None = None,
    stop: float | None = None,
    nan_policy: Literal["ffill", "scale"] = "ffill",
    nan_warn_threshold: float = 0.2,
) -> xr.DataArray:
    """Compute the straightness index of a path (D / L).

    The straightness index is the ratio of the Euclidean distance
    between the start and end positions of a trajectory to the
    total path length (D / L). Values range from 0 to 1, where 1
    indicates a perfectly straight path and 0 indicates the animal
    returned to its starting point. Returns NaN if path length is zero.

    Parameters
    ----------
    data : xarray.DataArray
        Position data with ``time`` and ``space`` dimensions.
    start : float, optional
        Start time. Defaults to first time coordinate.
    stop : float, optional
        Stop time. Defaults to last time coordinate.
    nan_policy : Literal["ffill", "scale"], optional
        How to handle NaN values. See ``compute_path_length``.
    nan_warn_threshold : float, optional
        Warn if NaN proportion exceeds this. Defaults to 0.2.

    Returns
    -------
    xarray.DataArray
        Straightness index per individual/keypoint, with ``time``
        and ``space`` dimensions removed.

    """
    validate_dims_coords(data, {"time": [], "space": []})
    data = data.sel(time=slice(start, stop))

    n_time = data.sizes["time"]
    if n_time < 2:
        raise logger.error(
            ValueError(
                "At least 2 time points are required to compute "
                f"path straightness, but {n_time} were found. "
                "Double-check the start and stop times."
            )
        )

    start_point = data.isel(time=0)
    end_point = data.isel(time=-1)
    distance = compute_norm(end_point - start_point)

    path_length = compute_path_length(
        data,
        nan_policy=nan_policy,
        nan_warn_threshold=nan_warn_threshold,
    )

    result = xr.where(path_length > 0, distance / path_length, np.nan)

    if "space" in result.dims:
        result = result.isel(space=0, drop=True)

    result.name = "straightness_index"
    result.attrs["units"] = "dimensionless"
    result.attrs["long_name"] = "Straightness Index (D/L)"
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
    # Skip first displacement segment (always 0) to not mess up the scaling
    displacement = compute_backward_displacement(data).isel(
        time=slice(1, None)
    )
    # count number of valid displacement segments per point track
    valid_segments = (~displacement.isnull()).all(dim="space").sum(dim="time")
    # compute proportion of valid segments per point track
    valid_proportion = valid_segments / (data.sizes["time"] - 1)
    # return scaled path length
    return compute_norm(displacement).sum(dim="time") / valid_proportion
