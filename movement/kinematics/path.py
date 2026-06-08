"""Compute metrics related to the length, straightness and complexity of paths.

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
from movement.utils.vector import compute_norm, compute_signed_angle_2d
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
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    nan_policy
        Policy to handle NaN (missing) values. Can be one of the ``"ffill"``
        or ``"scale"``. Defaults to ``"ffill"`` (forward fill).
        See Notes for more details on the two policies.
    nan_warn_threshold
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
    compute_path_straightness : A related metric that quantifies
        the straightness of a path.

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
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    nan_policy
        Policy to handle NaN (missing) values for the path length computation.
        Can be one of ``"ffill"`` or ``"scale"``. Defaults to ``"ffill"``
        (forward fill). See :func:`compute_path_length` for more details on
        the two policies.
    nan_warn_threshold
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
    compute_path_length : The underlying function used to
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


def compute_turning_angle(
    data: xr.DataArray,
    in_degrees: bool = False,
    min_step_length: float = 0.0,
) -> xr.DataArray:
    r"""Compute the turning angles between consecutive steps in a path.

    The turning angle at time ``t`` is the :func:`signed angle\
    <movement.utils.vector.compute_signed_angle_2d>` between two
    consecutive :func:`backward displacement\
    <movement.kinematics.compute_backward_displacement>` vectors
    at times ``t-1`` and ``t``.
    The returned angles are in radians, spanning the range :math:`(-\pi, \pi]`,
    unless ``in_degrees`` is set to ``True``.

    Parameters
    ----------
    data
        The input position data. Must contain ``time`` and ``space``
        dimensions. The ``space`` dimension must contain exactly the
        coordinates ``["x", "y"]`` (2D spatial data only).
    in_degrees
        If ``True``, return turning angles in degrees. Default is
        ``False`` (radians).
    min_step_length
        The minimum step length to consider for computing the turning
        angle. Any turning angle involving an incoming or outgoing step
        shorter than or equal to this value is set to ``NaN``. The
        default ``0.0`` only masks steps with exactly zero length,
        which means steps with near-zero lengths may still produce
        spurious angles. See Note 2 below.

    Returns
    -------
    xarray.DataArray
        Turning angles with the same shape as the input ``data``, but
        with the ``space`` dimension dropped.

    Notes
    -----
    1. **Time dimension length:** This function uses a ``shift``
       operation to preserve the original ``time`` dimension length.
       The first two time steps are always ``NaN``: the first because
       no previous step exists, and the second because a turning angle
       requires two steps (three positions). In other words, the turning angle
       at time step ``t`` is computed as the angle between the steps
       from ``t-2`` to ``t-1`` and from ``t-1`` to ``t``.
    2. **Positional jitter and small steps:** Tracking data
       often contains positional jitter, meaning a stationary animal
       may appear to make microscopic movements. With default parameters
       (``min_step_length=0.0``), these tiny, noisy movements will
       produce spurious, meaningless turning angles. It is highly
       recommended to set ``min_step_length`` to an appropriate threshold
       based on the tracking resolution and the animal's size in the scene.
       The value should be in the same units as the input position data
       (e.g. pixels, mm, etc.). Pre-smoothing the trajectory
       can also help reduce positional jitter.
    3. **NaN propagation:** ``NaN`` positions in the input propagate
       to ``NaN`` turning angles. A single missing position affects
       up to two turning angles (the incoming and outgoing steps).
       Use :func:`movement.filtering.interpolate_over_time` to fill
       positional gaps before computing turning angles if continuity
       is important.

    See Also
    --------
    movement.kinematics.compute_backward_displacement :
        The underlying function used to compute the displacement vectors.
    movement.utils.vector.compute_signed_angle_2d :
        The underlying function used to compute the signed angle
        between two consecutive displacement vectors.

    Examples
    --------
    >>> from movement.kinematics import compute_turning_angle

    Compute turning angles from the centroid trajectory of a poses
    dataset ``ds``:

    >>> centroid = ds.position.mean(dim="keypoint")
    >>> angles = compute_turning_angle(centroid)

    Compute in degrees, with a minimum step length of 3 pixels to filter out
    pose estimation jitter:

    >>> angles = compute_turning_angle(
    ...     centroid, in_degrees=True, min_step_length=3
    ... )

    """
    validate_dims_coords(
        data, {"time": [], "space": ["x", "y"]}, exact_coords=True
    )

    # Displacement arriving at each time step t.
    disp = compute_backward_displacement(data)

    # Turning angle at t = rotation needed to align step[t-1] onto step[t].
    turning = compute_signed_angle_2d(disp.shift(time=1), disp)

    # Mask turning angles involving steps smaller than min_step_length
    step_lengths = compute_norm(disp)
    invalid_steps = (step_lengths <= min_step_length) | (
        step_lengths.shift(time=1) <= min_step_length
    )
    turning = xr.where(invalid_steps, np.nan, turning)

    turning.attrs["units"] = "radians"

    if in_degrees:
        turning = np.rad2deg(turning)
        turning.attrs["units"] = "degrees"

    turning.name = "turning_angle"

    return turning


def compute_directional_change(
    data: xr.DataArray,
    in_degrees: bool = False,
    min_step_length: float = 0.0,
) -> xr.DataArray:
    r"""Compute the directional change (DC) per time step.

    The directional change at step :math:`i` is the absolute turning
    angle divided by the temporal interval spanning the two steps that
    define it [1]_:

    .. math::
        \mathrm{DC}_i = \frac{|\theta_i|}{\Delta t_i}

    where :math:`\theta_i` is the signed turning angle at step :math:`i`
    and :math:`\Delta t_i = t_i - t_{i-2}`.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    in_degrees : bool, optional
        If ``True``, the turning angles (and hence the directional
        change) are expressed in degrees rather than radians. Defaults
        to ``False``.
    min_step_length : float, optional
        The minimum step length used when computing turning angles.
        Steps shorter than or equal to this value produce ``NaN``
        turning angles, which propagate to ``NaN`` directional change
        values. The default ``0.0`` only masks steps with exactly
        zero length; near-zero steps from positional jitter may still
        produce spurious turning angles. See
        :func:`compute_turning_angle` for details.

    Returns
    -------
    xarray.DataArray
        Directional change values with the same dimensions as the
        input, except ``space`` is removed. Values are in radians per
        ``time`` unit (e.g. radians/second if ``time`` is in seconds),
        or degrees per ``time`` unit if ``in_degrees`` is ``True``.

    Notes
    -----
    **Boundary behaviour:** The first two time steps of the output are
    always ``NaN``, because a turning angle requires three consecutive
    positions (see :func:`compute_turning_angle`).

    References
    ----------
    .. [1] Kitamura, T. & Imafuku, M. (2015). Behavioural mimicry in
       flight path of Batesian intraspecific polymorphic butterfly
       *Papilio polytes*. *Proc. R. Soc. B* 282(1809).
       https://doi.org/10.1098/rspb.2015.0483

    See Also
    --------
    :func:`compute_turning_angle` :
        The underlying function used to compute turning angles.

    Examples
    --------
    >>> from movement.kinematics import compute_directional_change

    Compute directional change from the centroid trajectory of a
    poses dataset ``ds``:

    >>> centroid = ds.position.mean(dim="keypoint")
    >>> dc = compute_directional_change(centroid)

    Compute over a specific time window:

    >>> dc = compute_directional_change(centroid.sel(time=slice(0, 100)))

    Filter out pose-estimation jitter by setting ``min_step_length``:

    >>> dc = compute_directional_change(centroid, min_step_length=3)

    """
    data = _validate_time_points(data, "directional change")

    theta = compute_turning_angle(
        data, in_degrees=in_degrees, min_step_length=min_step_length
    )

    time_coord = data["time"]
    # Span the two steps that define theta_i (positions i-2..i), matching
    # compute_turning_angle's support so DC is correct for non-uniform time.
    dt = time_coord - time_coord.shift(time=2)

    dc = abs(theta) / dt
    dc.name = "directional_change"
    dc.attrs["long_name"] = "Directional Change"
    return dc


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
