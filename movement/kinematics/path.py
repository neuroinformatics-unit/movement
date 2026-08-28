"""Compute metrics related to the length, straightness and complexity of paths.

By 'path' we refer to the spatial trajectory of an individual over the
time span of the data. While these metrics can be computed based on any
set of keypoints, they are most meaningful when applied to a single
keypoint representing the individual's overall position (e.g., centroid).

When adding a new path metric, follow this convention for the
``nan_warn_threshold`` parameter: expose it only if the metric returns a
single scalar per path (e.g. ``compute_path_length``,
``compute_path_straightness``, ``compute_path_sinuosity`` and
``compute_path_emax``), where a high proportion of missing values can
silently skew that scalar. Do not expose it if the metric returns a
time-varying quantity (e.g. ``compute_directional_change`` and
``compute_path_deviation``), where missing values remain visible per
time step and propagate on their own.
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
    compute_path_sinuosity :
        A related turning-angle-based measure of path tortuosity.
    compute_path_emax :
        An alternative straightness measure derived from the
        turning-angle distribution which, unlike the :math:`D/L`
        ratio, does not depend on the number of steps in the path.

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


def compute_path_sinuosity(
    data: xr.DataArray,
    nan_warn_threshold: float = 0.2,
) -> xr.DataArray:
    r"""Compute the sinuosity of a path.

    Sinuosity (S) quantifies the tortuosity of a path by combining
    turning angle statistics with step-length variability. Higher
    values indicate more tortuous movement. A perfectly straight
    path has S = 0.

    The corrected sinuosity index (Eq. 8 in [1]_) is defined as:

    .. math::

        S = 2\left[\bar{p}\left(
            \frac{1+\bar{c}}{1-\bar{c}} + b^{2}
        \right)\right]^{-1/2}

    where :math:`\bar{p}` is the mean step length,
    :math:`\bar{c} = \tfrac{1}{n}\sum_{i=1}^{n}\cos(\phi_i)` is the mean
    cosine of turning angles, and
    :math:`b = \mathrm{SD}(p_i)\,/\,\bar{p}` is the coefficient of
    variation of step length.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    nan_warn_threshold : float, optional
        If any point track in the data has at least (:math:`\ge`)
        this proportion of values missing, a warning will be emitted.
        Defaults to ``0.2`` (20%).

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed sinuosity,
        with dimensions matching those of the input data,
        except ``time`` and ``space`` are removed.

    See Also
    --------
    compute_path_length : Total distance travelled along a path.
    compute_path_straightness : Net displacement divided by path length.
    compute_turning_angle : Step-wise turning angle along a path.
    compute_path_emax : Directional-persistence measure.

    Notes
    -----
    Step lengths are computed as the norm of backward displacement vectors
    via :func:`~movement.utils.vector.compute_norm` and
    :func:`~movement.kinematics.compute_backward_displacement`.
    Turning angles are computed via :func:`compute_turning_angle`.

    NaN positions propagate to NaN step lengths and turning angles;
    the statistics are then computed over the remaining valid samples.
    An entirely stationary track, or one with all NaN values,
    will produce NaN sinuosity.

    Sinuosity has units of :math:`1/\sqrt{\text{length}}`, so its
    numerical value depends on the position units of the input data.
    Values are not directly comparable across datasets recorded in
    different spatial units.

    References
    ----------
    .. [1] Benhamou, S. (2004). How to reliably estimate the tortuosity
       of an animal's path: straightness, sinuosity, or fractal dimension?
       *Journal of Theoretical Biology*, 229(2), 209-220.
       https://doi.org/10.1016/j.jtbi.2004.03.016

    Examples
    --------
    >>> from movement.kinematics import compute_path_sinuosity

    Compute sinuosity for the centroid trajectory of a poses dataset ``ds``:

    >>> centroid = ds.position.mean(dim="keypoint")
    >>> sinuosity = compute_path_sinuosity(centroid)

    Compute sinuosity over a specific time window:

    >>> sinuosity = compute_path_sinuosity(centroid.sel(time=slice(0, 100)))

    """
    data = _validate_time_points(
        data, metric_name="path sinuosity", min_points=3
    )

    _warn_about_nan_proportion(data, nan_warn_threshold)

    step_lengths = _segment_lengths(data)
    theta = compute_turning_angle(data)

    # Summary statistics (NaN-aware)
    mean_step_length = step_lengths.mean(dim="time", skipna=True)
    mean_cosine = xr.apply_ufunc(np.cos, theta).mean(dim="time", skipna=True)
    step_length_cv = (
        step_lengths.std(dim="time", skipna=True) / mean_step_length
    )

    # Benhamou 2004 Eq. 8
    angular_term = (1.0 + mean_cosine) / (1.0 - mean_cosine)
    result = (
        2.0 * (mean_step_length * (angular_term + step_length_cv**2)) ** -0.5
    )

    result.name = "sinuosity"
    result.attrs["long_name"] = "Path Sinuosity"

    return result


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
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    in_degrees
        If ``True``, the turning angles (and hence the directional
        change) are expressed in degrees rather than radians. Defaults
        to ``False``.
    min_step_length
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
    compute_turning_angle :
        The underlying function used to compute turning angles.
    compute_path_emax :
        A related path-straightness measure based on turning angles.

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


def compute_path_emax(
    data: xr.DataArray,
    in_spatial_units: bool = True,
    nan_warn_threshold: float = 0.2,
) -> xr.DataArray:
    r"""Compute the maximum expected displacement (:math:`E_{\max}`).

    :math:`E_{\max}` is a straightness measure that captures the
    directional persistence of a path. Intuitively, it is the maximum
    expected displacement of an animal navigating *without* an external
    directional reference (e.g. a compass or a landmark), given the
    observed distribution of its turning angles and step lengths [1]_.
    Larger values indicate straighter, more persistent paths; values
    close to zero indicate sinuous paths.

    Two variants are available. The dimensionless variant
    :math:`E_{\max}^{(a)}` depends only on the turning angles:

    .. math::
        E_{\max}^{(a)} = \frac{\bar{c}}{1 - \bar{c}}, \qquad
        \bar{c} = \overline{\cos\theta}

    where :math:`\theta` are the turning angles and :math:`\bar{c}` is
    their mean cosine. The variant :math:`E_{\max}^{(b)}` scales this by
    the mean step length :math:`\bar{p}` to express the result in the
    same spatial units as the input:

    .. math::
        E_{\max}^{(b)} = \bar{p} \, E_{\max}^{(a)}

    Parameters
    ----------
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    in_spatial_units
        If ``True`` (the default), return the dimensioned variant
        :math:`E_{\max}^{(b)}`, expressed in the same spatial units as
        ``data``. If ``False``, return the dimensionless variant
        :math:`E_{\max}^{(a)}`.
    nan_warn_threshold
        If any point track in the data has at least (:math:`\ge`)
        this proportion of values missing, a warning will be emitted.
        Defaults to 0.2 (20%).

    Returns
    -------
    xarray.DataArray
        The maximum expected displacement, with dimensions matching
        those of the input data, except ``time`` and ``space`` are
        removed. When ``in_spatial_units`` is ``True`` the values are in
        the same spatial units as ``data``; otherwise they are
        dimensionless.

    See Also
    --------
    compute_turning_angle :
        The underlying function used to compute the turning angles.
    compute_path_sinuosity :
        A related turning-angle-based measure of path tortuosity.
    compute_path_straightness :
        A related, path-length-based measure of straightness.

    Notes
    -----
    1. **Mean cosine of the turning angles.**
       :math:`\bar{c} = \overline{\cos\theta}` is the ``time`` average
       (ignoring ``NaN`` values) of the cosine of the turning angles
       :math:`\theta` returned by :func:`compute_turning_angle`. The first
       two time steps have no defined turning angle and so do not
       contribute.
    2. **Range.** :math:`E_{\max}^{(a)} \in [-0.5, \infty)`. While
       highly sinuous paths actually have values approaching 0,
       negative values specifically arise for trajectories with a
       systematic backward-turning bias where the mean cosine
       :math:`\bar{c}` is itself negative.
    3. **Straight paths.** As a path approaches a perfectly straight
       line, :math:`\bar{c} \to 1`, so :math:`1 - \bar{c} \to 0` and
       :math:`E_{\max} \to +\infty`. An infinite result is therefore the
       correct, expected output for a straight path.
    4. **Missing values.** Turning angles and step lengths that are
       ``NaN`` (e.g. from missing positions or stationary steps) are
       ignored when averaging. If every turning angle is ``NaN`` (e.g. a
       stationary track), the result is ``NaN``.

    References
    ----------
    .. [1] Cheung, A., Zhang, S., Stricker, C. & Srinivasan, M. V.
       (2007). Animal navigation: the difficulty of moving in a straight
       line. *Biological Cybernetics* 97(1), 47-61.
       https://doi.org/10.1007/s00422-007-0158-0

    Examples
    --------
    >>> from movement.kinematics import compute_path_emax

    Compute E_max from the centroid trajectory of a poses dataset ``ds``:

    >>> centroid = ds.position.mean(dim="keypoint")
    >>> emax = compute_path_emax(centroid)

    Return the dimensionless variant instead:

    >>> emax_a = compute_path_emax(centroid, in_spatial_units=False)

    Compute over a specific time window:

    >>> emax = compute_path_emax(centroid.sel(time=slice(0, 100)))

    """
    data = _validate_time_points(
        data, metric_name="maximum expected displacement", min_points=3
    )

    _warn_about_nan_proportion(data, nan_warn_threshold)

    theta = compute_turning_angle(data)
    mean_cosine = xr.apply_ufunc(np.cos, theta).mean(dim="time", skipna=True)

    emax = mean_cosine / (1 - mean_cosine)

    if in_spatial_units:
        emax = emax * _segment_lengths(data).mean(dim="time", skipna=True)

    emax.name = "maximum_expected_displacement"
    emax.attrs["long_name"] = "Maximum Expected Displacement"
    return emax


def compute_path_deviation(
    data: xr.DataArray,
) -> xr.DataArray:
    r"""Compute deviation from the straight-line path at each time point.

    For each time point :math:`t`, the path deviation is the perpendicular
    (unsigned) distance between the position :math:`P(t)` and the infinite
    straight line passing through the first and last valid positions in
    the data, denoted :math:`A` and :math:`B` respectively. Zero means the
    position lies exactly on the straight line; larger values indicate greater
    lateral excursion.

    Formally, let :math:`\mathbf{u} = B - A` be the chord vector and
    :math:`\hat{\mathbf{u}} = \mathbf{u} / \|\mathbf{u}\|` its unit vector.
    The deviation at time :math:`t` is:

    .. math::

        d(t) = \left\|
            (P(t) - A) -
            \left[(P(t) - A) \cdot \hat{\mathbf{u}}\right] \hat{\mathbf{u}}
        \right\|

    This is the norm of the component of :math:`P(t) - A` that is
    perpendicular to the chord, and is equivalent to the distance from
    :math:`P(t)` to the infinite line through :math:`A` and :math:`B`.
    The formulation is dimension-agnostic (works for 2D and 3D data).

    Parameters
    ----------
    data
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the perpendicular deviation from
        the chord at each time point, with dimensions matching those of
        the input data, except ``space`` is removed. Values are in the
        same spatial units as the input.

    Raises
    ------
    ValueError
        If fewer than 2 time points exist in the data, or if
        the chord length is zero (i.e. ``A == B``) for *all* tracks.
        If the chord length is zero for *some* (but not all) tracks, a
        warning is issued and those tracks will have ``NaN`` deviation.


    See Also
    --------
    compute_path_length : Total distance travelled along the path.
    compute_path_straightness : Ratio of chord length to path length.

    Examples
    --------
    >>> from movement.kinematics import compute_path_deviation

    Compute per-frame path deviation from the centroid trajectory of a
    poses dataset ``ds``:

    >>> centroid = ds.position.mean(dim="keypoint")
    >>> deviation = compute_path_deviation(centroid)

    Compute the maximum lateral excursion over the trajectory:

    >>> max_deviation = deviation.max(dim="time")

    Compute the mean deviation over a specific time window:

    >>> mean_deviation = compute_path_deviation(
    ...     centroid.sel(time=slice(10, 50))
    ... ).mean(dim="time")

    """
    data = _validate_time_points(data, "path deviation")

    anchored = data.ffill(dim="time").bfill(dim="time")
    A = anchored.isel(time=0)
    B = anchored.isel(time=-1)

    chord = B - A
    chord_length = compute_norm(chord)

    degenerate = chord_length == 0
    if degenerate.all():
        raise logger.error(
            ValueError(
                "Path deviation is undefined because the start and end "
                "positions are identical for all tracks."
            )
        )
    if degenerate.any():
        stacked = degenerate.stack(tracks=list(degenerate.dims))
        bad_tracks = stacked.sel(tracks=stacked).coords["tracks"].values
        warnings.warn(
            "Path deviation is undefined for tracks where the start and end "
            "positions are identical. The following tracks will return NaN: "
            f"{bad_tracks}",
            UserWarning,
            stacklevel=2,
        )

    chord_unit = chord / chord_length
    p_minus_a = data - A
    scalar_proj = (p_minus_a * chord_unit).sum(dim="space")
    vector_proj = scalar_proj * chord_unit
    rejection = p_minus_a - vector_proj

    deviation = compute_norm(rejection)
    deviation.name = "path_deviation"
    deviation.attrs["long_name"] = "Path Deviation"
    return deviation


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
    min_points: int = 2,
) -> xr.DataArray:
    """Validate dims/coords and require at least ``min_points`` time points.

    Parameters
    ----------
    data : xarray.DataArray
        Position data with ``time`` and ``space`` dimensions.
    metric_name : str
        Used in the error message when there are fewer than ``min_points``
        time points.
    min_points : int, optional
        The minimum number of time points required. Defaults to 2.

    Returns
    -------
    xarray.DataArray
        The validated data.

    """
    validate_dims_coords(data, {"time": [], "space": []})
    n_time = data.sizes["time"]
    if n_time < min_points:
        raise logger.error(
            ValueError(
                f"At least {min_points} time points are required to compute "
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
