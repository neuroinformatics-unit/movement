"""Compute path-level metrics such as path length and straightness.

By 'path' we refer to the spatial trajectory of an individual over the
time span of the data. While these metrics can be computed based on any
set of keypoints, they are most meaningful when applied to a single
keypoint representing the individual's overall position (e.g., centroid).
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
    nan_policy: Literal["ffill", "scale"] = "ffill",
    nan_warn_threshold: float = 0.2,
) -> xr.DataArray:
    r"""Compute the length of a path travelled.

    The path length is defined as the sum of the norms (magnitudes) of the
    displacement vectors between consecutive time points in the data.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
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

    **Sampling rate sensitivity ('coastline paradox'):**
    The measured path length is sensitive to the temporal sampling rate
    (i.e., frames per second) of the tracking data. Higher sampling rates
    capture finer micro-movements and tracking jitter, which inherently
    increases the total measured path length. Exercise caution when comparing
    path lengths across datasets with different temporal resolutions.

    See Also
    --------
    :func:`compute_path_straightness`

    Examples
    --------
    >>> from movement.kinematics import compute_path_length

    Compute the path length from the centroid trajectory of a poses
    dataset ``ds``:

    >>> centroid = ds.position.mean(dim="keypoint")
    >>> length = compute_path_length(centroid)

    Compute path length over a specific time window:

    >>> length = compute_path_length(centroid.sel(time=slice(0, 100)))

    Use the scale policy to handle missing values:

    >>> length = compute_path_length(centroid, nan_policy="scale")

    """
    data = _validate_time_points(data, "path length")
    return _path_length(data, nan_policy, nan_warn_threshold)


def compute_path_straightness(
    data: xr.DataArray,
    nan_policy: Literal["ffill", "scale"] = "ffill",
    nan_warn_threshold: float = 0.2,
) -> xr.DataArray:
    r"""Compute the straightness index of a path :math:`(D/L)`.

    The straightness index is the ratio of the Euclidean distance :math:`D`
    between the first and last valid positions of a trajectory to the
    total path length :math:`L`. Values range from 0 to 1, where 1
    indicates a perfectly straight path and 0 indicates the animal
    returned to its starting point. Returns NaN if the path length is zero.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    nan_policy : Literal["ffill", "scale"], optional
        Policy to handle NaN (missing) values for the path length computation.
        Can be one of ``"ffill"`` or ``"scale"``. Defaults to ``"ffill"``
        (forward fill). See :func:`compute_path_length` for more details on
        the two policies.
    nan_warn_threshold : float, optional
        If any point track in the data has at least (:math:`\ge`)
        this proportion of values missing, a warning will be emitted.
        Defaults to 0.2 (20%). Directly passed to :func:`compute_path_length`.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed straightness index,
        with dimensions matching those of the input data,
        except ``time`` and ``space`` are removed.

    Notes
    -----
    The Euclidean distance :math:`D`, also known as the "straight-line" or
    "beeline" distance, is calculated using the first and last valid (non-NaN)
    spatial coordinates in the data. This ensures that missing data at the
    first or last time points do not nullify the result, provided there are
    valid observed positions in between.

    Note that the total path length (L), and therefore the straightness index,
    is sensitive to the temporal sampling  rate (i.e. frames per second),
    as described in the Notes of :func:`compute_path_length`.

    See Also
    --------
    :func:`compute_path_length` : The underlying function used to
        compute the path length :math:`L`.

    Examples
    --------
    >>> from movement.kinematics import compute_path_straightness

    Compute the straightness index from the centroid trajectory of a
    poses dataset ``ds``:

    >>> centroid = ds.position.mean(dim="keypoint")
    >>> si = compute_path_straightness(centroid)

    Compute straightness over a specific time window:

    >>> si = compute_path_straightness(centroid.sel(time=slice(0, 100)))

    """
    data = _validate_time_points(data, "path straightness")
    path_length = _path_length(data, nan_policy, nan_warn_threshold)
    # Compute D/L ratio, avoiding division by zero
    result = _path_distance(data) / path_length.where(path_length > 0)
    result.name = "straightness_index"
    result.attrs["long_name"] = "Path Straightness Index"
    return result


def compute_roaming_entropy(
    data: xr.DataArray,
    bins: int | tuple[int, int] = 30,
    normalise: bool = True,
) -> xr.DataArray:
    r"""Compute the roaming entropy of a path.

    Roaming entropy quantifies how broadly and uniformly an individual
    explores the 2D space over the course of a trajectory. It is defined as
    the Shannon entropy of the spatial occupancy distribution obtained by
    binning the positions onto a regular 2D grid. Given a grid of :math:`N`
    bins, where :math:`p_i` is the proportion of (non-NaN) time points spent
    in bin :math:`i`, the roaming entropy :math:`H` is

    .. math::
        H = -\sum_{i=1}^{N} p_i \ln p_i,

    with the convention :math:`0 \ln 0 = 0`. The value ranges from
    :math:`0` (the individual stays within a single bin) to :math:`\ln N`
    (the individual visits all bins equally). By default the result is
    normalised to the range :math:`[0, 1]` by dividing by :math:`\ln N`.

    The metric is invariant to the *order* in which bins are visited, so it
    captures *where* the individual went rather than *how* it got there. As
    such, it complements speed- and distance-based metrics such as
    :func:`compute_path_length`.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time`` and
        ``space`` (containing the ``"x"`` and ``"y"`` coordinates) as
        required dimensions.
    bins : int or tuple of int, optional
        The number of bins along the ``x`` and ``y`` axes used to build the
        occupancy grid. If an integer is provided, the same number of bins is
        used for both axes. If a tuple ``(nx, ny)`` is provided, ``nx`` and
        ``ny`` bins are used along the ``x`` and ``y`` axes, respectively.
        Defaults to 30. See Notes on the sensitivity of the metric to this
        parameter.
    normalise : bool, optional
        If ``True`` (default), the entropy is divided by :math:`\ln N`
        (where :math:`N` is the total number of bins), scaling the result to
        the range :math:`[0, 1]`. If ``False``, the unnormalised entropy in
        nats is returned, ranging in :math:`[0, \ln N]`.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed roaming entropy, with
        dimensions matching those of the input data, except ``time`` and
        ``space`` are removed (i.e. a scalar per individual and/or keypoint).
        Tracks with no valid (non-NaN) positions yield NaN.

    Notes
    -----
    The occupancy grid spans the full extent of the valid positions in
    ``data``, i.e. its bounds are derived from the global minimum and maximum
    of the ``x`` and ``y`` coordinates across all time points, individuals and
    keypoints. Using a shared grid in this way makes the resulting entropies
    directly comparable across individuals and keypoints within the same
    dataset.

    The roaming entropy is sensitive to the choice of ``bins``: a finer grid
    (more bins) increases the maximum attainable entropy and the metric's
    sensitivity to small movements, while a coarser grid emphasises broad-scale
    exploration. When comparing values across datasets, ensure that the same
    ``bins`` (and, ideally, a comparable spatial extent) are used.

    References
    ----------
    .. [1] Freund, J., Brandmaier, A. M., Lewejohann, L., Kirste, I.,
       Kritzler, M., Krüger, A., Sachser, N., Lindenberger, U., & Kempermann,
       G. (2013). Emergence of individuality in genetically identical mice.
       Science, 340(6133), 756-759. https://doi.org/10.1126/science.1235294

    See Also
    --------
    :func:`compute_path_length` : A complementary, distance-based path metric.
    movement.plots.plot_occupancy : Plot the 2D occupancy histogram.

    Examples
    --------
    >>> from movement.kinematics import compute_roaming_entropy

    Compute the (normalised) roaming entropy from the centroid trajectory of
    a poses dataset ``ds``:

    >>> centroid = ds.position.mean(dim="keypoint")
    >>> entropy = compute_roaming_entropy(centroid)

    Use a coarser grid and return the unnormalised entropy in nats:

    >>> entropy = compute_roaming_entropy(centroid, bins=10, normalise=False)

    """
    validate_dims_coords(data, {"time": [], "space": ["x", "y"]})
    data = data.sel(space=["x", "y"])
    bins_xy = (bins, bins) if isinstance(bins, int) else tuple(bins)
    if len(bins_xy) != 2 or any(int(b) < 1 for b in bins_xy):
        raise logger.error(
            ValueError(
                "bins must be a positive integer or a tuple of two positive "
                f"integers, but got {bins}."
            )
        )
    # Derive a shared grid extent from the global span of valid positions, so
    # that entropies are comparable across individuals and keypoints.
    x_values = data.sel(space="x").values
    y_values = data.sel(space="y").values
    if np.all(np.isnan(x_values)) or np.all(np.isnan(y_values)):
        # No valid positions anywhere: every track's entropy is undefined.
        return xr.full_like(
            data.isel(space=0, drop=True).isel(time=0, drop=True), np.nan
        ).rename("roaming_entropy")
    hist_range = [
        [float(np.nanmin(x_values)), float(np.nanmax(x_values))],
        [float(np.nanmin(y_values)), float(np.nanmax(y_values))],
    ]
    result = xr.apply_ufunc(
        _roaming_entropy,
        data,
        input_core_dims=[["time", "space"]],
        kwargs={
            "bins": list(bins_xy),
            "hist_range": hist_range,
            "normalise": normalise,
        },
        vectorize=True,
    )
    result.name = "roaming_entropy"
    result.attrs["long_name"] = "Roaming Entropy"
    return result


def _roaming_entropy(
    track: np.ndarray,
    bins: list[int],
    hist_range: list[list[float]],
    normalise: bool,
) -> float:
    """Compute the roaming entropy for a single ``(time, space)`` track.

    See :func:`compute_roaming_entropy` for parameter details. Returns NaN if
    the track contains no valid (non-NaN) positions.
    """
    x = track[:, 0]
    y = track[:, 1]
    valid = ~(np.isnan(x) | np.isnan(y))
    if not valid.any():
        return np.nan
    counts, _, _ = np.histogram2d(
        x[valid], y[valid], bins=bins, range=hist_range
    )
    counts = counts.ravel()
    total = counts.sum()
    if total == 0:
        return np.nan
    # Restrict to occupied bins (0 * ln 0 = 0 by convention).
    proportions = counts[counts > 0] / total
    entropy = float(-np.sum(proportions * np.log(proportions)))
    if normalise:
        n_bins = counts.size
        entropy = entropy / np.log(n_bins) if n_bins > 1 else 0.0
    return entropy


def _validate_time_points(
    data: xr.DataArray,
    metric_name: str,
) -> xr.DataArray:
    """Validate dims/coords and require at least 2 time points.

    Parameters
    ----------
    data : xarray.DataArray
        Position data with ``time`` and ``space`` dimensions.
    metric_name : str
        Used in the error message when there are fewer than 2 time points.

    Returns
    -------
    xarray.DataArray
        The validated data.

    """
    validate_dims_coords(data, {"time": [], "space": []})
    n_time = data.sizes["time"]
    if n_time < 2:
        raise logger.error(
            ValueError(
                "At least 2 time points are required to compute "
                f"{metric_name}, but {n_time} were found."
            )
        )
    return data


def _segment_lengths(data: xr.DataArray) -> xr.DataArray:
    """Compute Euclidean distances between consecutive time points.

    The first entry of backward displacement is always zero (no previous
    point), so it is dropped before computing the norm.
    """
    segments = compute_backward_displacement(data).isel(time=slice(1, None))
    return compute_norm(segments)


def _path_distance(data: xr.DataArray) -> xr.DataArray:
    """Compute Euclidean distance between the first and last valid positions.

    Also known as the "straight-line" or "beeline" distance.
    Uses forward and backward filling along the time dimension to ensure
    the distance is calculated between the first and last observed locations,
    preventing NaNs at the first or last time points from nullifying the
    entire calculation.
    """
    anchored_data = data.ffill(dim="time").bfill(dim="time")
    distance = compute_norm(
        anchored_data.isel(time=-1) - anchored_data.isel(time=0)
    )
    return distance


def _path_length(
    data: xr.DataArray,
    nan_policy: Literal["ffill", "scale"],
    nan_warn_threshold: float,
) -> xr.DataArray:
    """Compute path length on already-validated data.

    See :func:`compute_path_length` for parameter details.
    """
    _warn_about_nan_proportion(data, nan_warn_threshold)
    if nan_policy == "ffill":
        result = _segment_lengths(data.ffill(dim="time")).sum(
            dim="time", min_count=1
        )
    elif nan_policy == "scale":
        lengths = _segment_lengths(data)
        valid_segments = (~lengths.isnull()).sum(dim="time")
        valid_proportion = valid_segments / (data.sizes["time"] - 1)
        result = lengths.sum(dim="time") / valid_proportion
    else:
        raise logger.error(
            ValueError(
                f"Invalid value for nan_policy: {nan_policy}. "
                "Must be one of 'ffill' or 'scale'."
            )
        )
    result.name = "path_length"
    result.attrs["long_name"] = "Path Length"
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
    exceeds_threshold = n_nans >= data.sizes["time"] * nan_warn_threshold
    if not exceeds_threshold.any():
        return
    track_dims = [d for d in data.dims if d not in ("time", "space")]
    stacked = data.stack(tracks=track_dims)
    mask = exceeds_threshold.stack(tracks=track_dims)
    data_to_warn_about = stacked.sel(tracks=mask).unstack("tracks")
    warnings.warn(
        "The result may be unreliable for point tracks with many "
        "missing values. The following tracks have at least "
        f"{nan_warn_threshold * 100:.3} % NaN values:\n"
        f"{report_nan_values(data_to_warn_about)}",
        UserWarning,
        stacklevel=2,
    )
