"""Compute collective behavior metrics for multi-animal groups.

This module provides functions to quantify group-level dynamics in
multi-animal tracking data, including alignment (polarization), rotational
coherence (milling), spatial dispersion (group spread), leader-follower
relationships (leadership), and pairwise relative-motion metrics
(egocentric angle, approach/tangent velocity).
"""

from collections.abc import Hashable

import numpy as np
import xarray as xr

from movement.utils.logging import logger
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


def _position_for_individuals(position: xr.DataArray) -> xr.DataArray:
    """Return position averaged over keypoints if the dim is present.

    Parameters
    ----------
    position
        Position data that may contain a ``keypoints`` dimension.

    Returns
    -------
    xarray.DataArray
        Position with dims ``(time, space, individuals)``.
    """
    if "keypoints" in position.dims:
        return position.mean(dim="keypoints", skipna=True)
    return position


def _compute_heading_vectors(
    position: xr.DataArray,
    heading_keypoints: tuple[Hashable, Hashable] | None,
) -> xr.DataArray:
    """Compute heading vectors with dims ``(time, space, individuals)``.

    Parameters
    ----------
    position
        Position data with at least ``time``, ``space``, and
        ``individuals`` dimensions.
    heading_keypoints
        If not ``None``, heading is computed as the vector from
        ``heading_keypoints[0]`` to ``heading_keypoints[1]``.
        Otherwise velocity-based heading is used.

    Returns
    -------
    xarray.DataArray
        Heading vectors with dims ``(time, space, individuals)``.
    """
    if heading_keypoints is not None:
        validate_dims_coords(
            position,
            {"keypoints": list(heading_keypoints)},
        )
        heading = position.sel(
            keypoints=heading_keypoints[1], drop=True
        ) - position.sel(keypoints=heading_keypoints[0], drop=True)
    else:
        pos = _position_for_individuals(position)
        heading = pos.differentiate("time")
    return heading


def compute_polarization(
    position: xr.DataArray,
    heading_keypoints: tuple[Hashable, Hashable] | None = None,
) -> xr.DataArray:
    r"""Compute group polarization: alignment of heading directions.

    Polarization quantifies whether individuals are heading in the same
    direction. Values near 1 indicate high alignment (e.g., a migrating
    flock), while values near 0 indicate random orientations (e.g., a
    dispersed swarm).

    Parameters
    ----------
    position
        Position data with dimensions ``(time, space, individuals)`` or
        ``(time, space, keypoints, individuals)``.
    heading_keypoints
        A pair ``(from_keypoint, to_keypoint)`` defining the heading
        direction, e.g., ``("tail", "nose")``. If ``None`` (default),
        velocity-based heading is computed via numerical differentiation.

    Returns
    -------
    xarray.DataArray
        Polarization values with dimension ``(time,)``, in range
        :math:`[0, 1]`. ``NaN`` when no valid individual is present.

    Notes
    -----
    The polarization order parameter is:

    .. math::

        \Phi = \frac{1}{N} \left| \sum_{i=1}^{N} \hat{h}_i \right|

    where :math:`\hat{h}_i` is the unit heading vector of individual
    :math:`i` and :math:`N` is the number of valid (non-NaN,
    non-stationary) individuals at each time step.

    References
    ----------
    .. [1] Couzin, I.D., Krause, J., James, R., Ruxton, G.D. and
       Franks, N.R. (2002). Collective memory and spatial sorting in
       animal groups. *Journal of Theoretical Biology*, 218(1), 1–11.

    Examples
    --------
    Keypoint-based polarization (tail → nose heading):

    >>> pol = compute_polarization(
    ...     ds.position, heading_keypoints=("tail", "nose")
    ... )

    Velocity-based polarization:

    >>> pol = compute_polarization(ds.position)
    """
    validate_dims_coords(
        position, {"time": [], "space": [], "individuals": []}
    )
    heading = _compute_heading_vectors(position, heading_keypoints)
    # heading: (time, space, individuals)

    heading_norm = compute_norm(heading)  # (time, individuals)
    valid = (~heading.isnull().any(dim="space")) & (heading_norm > 0)
    n_valid = valid.sum(dim="individuals").astype(float)

    unit_heading = (heading / heading_norm).where(valid)
    sum_heading = unit_heading.sum(dim="individuals", skipna=True)
    norm_sum = compute_norm(sum_heading)

    polarization = xr.where(n_valid > 0, norm_sum / n_valid, np.nan)
    polarization.name = "polarization"
    return polarization


def compute_milling(
    position: xr.DataArray,
    heading_keypoints: tuple[Hashable, Hashable] | None = None,
) -> xr.DataArray:
    r"""Compute group milling order: rotational coherence around a center.

    Milling quantifies whether individuals rotate collectively around a
    common center point. Values near 1 indicate coherent rotation (all
    individuals moving clockwise or all counterclockwise), while values
    near 0 indicate mixed or no rotation.

    Parameters
    ----------
    position
        Position data with dimensions ``(time, space, individuals)`` or
        ``(time, space, keypoints, individuals)``. Must have 2D space
        coordinates ``["x", "y"]``.
    heading_keypoints
        A pair ``(from_keypoint, to_keypoint)`` defining the heading
        direction. If ``None`` (default), velocity-based heading is used.

    Returns
    -------
    xarray.DataArray
        Milling values with dimension ``(time,)``, in range
        :math:`[0, 1]`. ``NaN`` when angular momentum is uniformly zero.

    Notes
    -----
    The milling order parameter is:

    .. math::

        M = \frac{\left|\sum_{i=1}^{N} l_i\right|}{\sum_{i=1}^{N} |l_i|}

    where :math:`l_i = \mathbf{r}_i \times \mathbf{v}_i` is the scalar
    angular momentum of individual :math:`i`, :math:`\mathbf{r}_i` is
    the position relative to the group centroid, and
    :math:`\mathbf{v}_i` is the heading vector.

    References
    ----------
    .. [1] Couzin, I.D., Krause, J., James, R., Ruxton, G.D. and
       Franks, N.R. (2002). Collective memory and spatial sorting in
       animal groups. *Journal of Theoretical Biology*, 218(1), 1–11.

    Examples
    --------
    >>> mill = compute_milling(ds.position, heading_keypoints=("tail", "nose"))
    """
    validate_dims_coords(
        position, {"time": [], "space": [], "individuals": []}
    )
    validate_dims_coords(position, {"space": ["x", "y"]}, exact_coords=True)
    pos = _position_for_individuals(position)
    heading = _compute_heading_vectors(position, heading_keypoints)

    centroid = pos.mean(dim="individuals", skipna=True)  # (time, space)
    r = pos - centroid  # (time, space, individuals)

    # 2D cross product scalar: r × v = r_x*v_y - r_y*v_x
    r_x = r.sel(space="x", drop=True)
    r_y = r.sel(space="y", drop=True)
    v_x = heading.sel(space="x", drop=True)
    v_y = heading.sel(space="y", drop=True)
    L = r_x * v_y - r_y * v_x  # (time, individuals)

    valid = ~(
        pos.isnull().any(dim="space") | heading.isnull().any(dim="space")
    )
    L_valid = L.where(valid)

    sum_L_abs = np.abs(L_valid.sum(dim="individuals", skipna=True))
    sum_abs_L = np.abs(L_valid).sum(dim="individuals", skipna=True)

    milling = xr.where(sum_abs_L > 0, sum_L_abs / sum_abs_L, np.nan)
    milling.name = "milling"
    return milling


def compute_group_spread(
    position: xr.DataArray,
    method: str = "radius_of_gyration",
) -> xr.DataArray:
    r"""Compute spatial dispersion of the group.

    Group spread measures how spread out individuals are relative to
    the group centroid. Larger values indicate a more dispersed group.

    Parameters
    ----------
    position
        Position data with dimensions ``(time, space, individuals)`` or
        ``(time, space, keypoints, individuals)``.
    method
        Method for computing dispersion. Currently only
        ``"radius_of_gyration"`` is supported (default).

    Returns
    -------
    xarray.DataArray
        Group spread values with dimension ``(time,)``, in spatial
        units. ``NaN`` when fewer than 2 valid individuals are present.

    Notes
    -----
    The radius of gyration is:

    .. math::

        R_g = \sqrt{\frac{1}{N} \sum_{i=1}^{N}
        \left|\mathbf{r}_i - \mathbf{r}_\mathrm{cm}\right|^2}

    where :math:`\mathbf{r}_i` is the position of individual :math:`i`
    and :math:`\mathbf{r}_\mathrm{cm}` is the group centroid.

    Examples
    --------
    >>> spread = compute_group_spread(ds.position)
    """
    validate_dims_coords(
        position, {"time": [], "space": [], "individuals": []}
    )
    if method != "radius_of_gyration":
        raise logger.error(
            ValueError(
                f"Unsupported method '{method}'. "
                "Currently only 'radius_of_gyration' is supported."
            )
        )

    pos = _position_for_individuals(position)
    centroid = pos.mean(dim="individuals", skipna=True)  # (time, space)
    diff = pos - centroid  # (time, space, individuals)
    dist_sq = (diff**2).sum(dim="space")  # (time, individuals)

    valid = ~pos.isnull().any(dim="space")
    n_valid = valid.sum(dim="individuals").astype(float)
    sum_dist_sq = dist_sq.where(valid).sum(dim="individuals", skipna=True)

    spread = np.sqrt(sum_dist_sq / n_valid)
    spread = xr.where(n_valid >= 2, spread, np.nan)
    spread.name = "group_spread"
    return spread


def compute_leadership(
    position: xr.DataArray,
    max_lag: int = 30,
) -> xr.DataArray:
    r"""Identify leader-follower relationships via velocity cross-correlation.

    For each ordered pair of individuals ``(i, j)``, finds the time lag
    :math:`\tau` at which their speed cross-correlation is maximised.
    A positive lag means individual ``i`` leads individual ``j``.

    Parameters
    ----------
    position
        Position data with dimensions ``(time, space, individuals)`` or
        ``(time, space, keypoints, individuals)``.
    max_lag
        Maximum lag in frames to search (searches
        :math:`\tau \in [-\text{max\_lag}, +\text{max\_lag}]`).
        Default is 30.

    Returns
    -------
    xarray.DataArray
        An array with dimensions
        ``(individuals, individuals_other, metric)`` where the
        ``metric`` coordinate has values ``["correlation", "lag"]``:

        - ``correlation``: Pearson correlation coefficient at the
          optimal lag.
        - ``lag``: Optimal lag in frames. Positive means
          ``individuals`` leads ``individuals_other``.

        Diagonal entries (self-pairs) are ``NaN``.

    Notes
    -----
    For each pair :math:`(i, j)`, the optimal lag is:

    .. math::

        \tau^* = \arg\max_{\tau} \,
        \mathrm{corr}\!\left(s_i(t),\, s_j(t + \tau)\right)

    where :math:`s_i(t)` is the speed of individual :math:`i` at time
    :math:`t`. Computational complexity is
    :math:`O(N^2 \times T \times L)` where :math:`L = \text{max\_lag}`.

    References
    ----------
    .. [1] Nagy, M., Ákos, Z., Biro, D. and Vicsek, T. (2010).
       Hierarchical group dynamics in pigeon flocks. *Nature*,
       464(7290), 890–893.

    Examples
    --------
    >>> lead = compute_leadership(ds.position, max_lag=10)
    >>> corr = lead.sel(metric="correlation")
    >>> lag  = lead.sel(metric="lag")
    """
    validate_dims_coords(
        position, {"time": [], "space": [], "individuals": []}
    )
    if not isinstance(max_lag, int) or max_lag <= 0:
        raise logger.error(
            ValueError(
                f"max_lag must be a positive integer, but got {max_lag!r}."
            )
        )

    pos = _position_for_individuals(position)
    velocity = pos.differentiate("time")
    speed = compute_norm(velocity)  # (time, individuals)

    individuals = position.coords["individuals"].values
    n_ind = len(individuals)
    corr_arr = np.full((n_ind, n_ind), np.nan)
    lag_arr = np.full((n_ind, n_ind), np.nan)

    speed_np = speed.values  # shape (n_time, n_individuals)

    for i in range(n_ind):
        for j in range(n_ind):
            if i == j:
                continue
            vi = speed_np[:, i]
            vj = speed_np[:, j]

            best_corr = -np.inf
            best_lag = 0

            for lag in range(-max_lag, max_lag + 1):
                if lag > 0:
                    vi_seg = vi[:-lag]
                    vj_seg = vj[lag:]
                elif lag < 0:
                    vi_seg = vi[-lag:]
                    vj_seg = vj[:lag]
                else:
                    vi_seg = vi
                    vj_seg = vj

                mask = ~(np.isnan(vi_seg) | np.isnan(vj_seg))
                if mask.sum() < 3:
                    continue

                vi_v = vi_seg[mask]
                vj_v = vj_seg[mask]
                if vi_v.std() < 1e-10 or vj_v.std() < 1e-10:
                    continue

                corr = float(np.corrcoef(vi_v, vj_v)[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

            if np.isfinite(best_corr):
                corr_arr[i, j] = best_corr
                lag_arr[i, j] = best_lag

    result = xr.DataArray(
        np.stack([corr_arr, lag_arr], axis=-1),
        dims=["individuals", "individuals_other", "metric"],
        coords={
            "individuals": individuals,
            "individuals_other": individuals,
            "metric": ["correlation", "lag"],
        },
    )
    result.name = "leadership"
    return result


def compute_egocentric_angle(
    position: xr.DataArray,
    heading_keypoints: tuple[Hashable, Hashable] | None = None,
    in_degrees: bool = False,
) -> xr.DataArray:
    r"""Compute the egocentric angle from each individual to every other.

    For each focal–other pair, the egocentric angle is the signed angle
    between the focal individual's heading vector and the vector pointing
    from the focal individual to the other. Positive angles indicate the
    other individual is to the left of the focal's heading direction
    (counter-clockwise in standard image coordinates).

    Parameters
    ----------
    position
        Position data with dimensions ``(time, space, individuals)`` or
        ``(time, space, keypoints, individuals)``. Must have 2D space
        coordinates ``["x", "y"]``.
    heading_keypoints
        A pair ``(from_keypoint, to_keypoint)`` defining the heading.
        If ``None`` (default), velocity-based heading is used.
    in_degrees
        If ``True``, return angles in degrees. Default is ``False``
        (radians).

    Returns
    -------
    xarray.DataArray
        Egocentric angles with dimensions
        ``(time, individuals, individuals_other)``, in radians (or
        degrees if ``in_degrees=True``). Range is :math:`(-\pi, \pi]`
        radians. Diagonal entries (self-pairs) are ``NaN``.

    Notes
    -----
    The egocentric angle :math:`\alpha_{ij}` of individual :math:`j`
    in the reference frame of individual :math:`i` is:

    .. math::

        \alpha_{ij} = \mathrm{atan2}
        \!\left(\hat{h}_i \times \mathbf{d}_{ij},\;
        \hat{h}_i \cdot \mathbf{d}_{ij}\right)

    where :math:`\hat{h}_i` is the unit heading vector of individual
    :math:`i` and :math:`\mathbf{d}_{ij} = \mathbf{r}_j - \mathbf{r}_i`
    is the vector from :math:`i` to :math:`j`.

    References
    ----------
    .. [1] Cheng, X. et al. (2025). Multimodal analysis of collective
       behavior. *Nature Methods*.

    Examples
    --------
    >>> angles = compute_egocentric_angle(
    ...     ds.position, heading_keypoints=("tail", "nose")
    ... )
    >>> angles_deg = compute_egocentric_angle(ds.position, in_degrees=True)
    """
    validate_dims_coords(
        position, {"time": [], "space": [], "individuals": []}
    )
    validate_dims_coords(position, {"space": ["x", "y"]}, exact_coords=True)
    heading = _compute_heading_vectors(position, heading_keypoints)
    pos = _position_for_individuals(position)

    # Vector from each focal individual i to each other individual j:
    # vec_to_other[..., i, j] = pos[j] - pos[i]
    pos_other = pos.rename({"individuals": "individuals_other"})
    vec_to_other = pos_other - pos  # (time, space, individuals, individuals_other)

    # Unit heading (normalize per individual)
    h_norm = compute_norm(heading)  # (time, individuals)
    h_x = heading.sel(space="x", drop=True) / h_norm
    h_y = heading.sel(space="y", drop=True) / h_norm

    v_x = vec_to_other.sel(space="x", drop=True)
    v_y = vec_to_other.sel(space="y", drop=True)

    # Signed angle: atan2(h × d, h · d)
    cross = h_x * v_y - h_y * v_x  # (time, individuals, individuals_other)
    dot = h_x * v_x + h_y * v_y

    angle = np.arctan2(cross, dot)

    # Map -pi to pi (arctan2 returns (-pi, pi], fix edge case)
    angle = xr.where(angle <= -np.pi, np.pi, angle)

    # Self-pairs: vector is zero → set to NaN
    vec_norm_sq = v_x**2 + v_y**2
    angle = xr.where(vec_norm_sq > 0, angle, np.nan)

    angle = angle.transpose("time", "individuals", "individuals_other")

    if in_degrees:
        angle = np.rad2deg(angle)

    angle.name = "egocentric_angle"
    return angle


def compute_approach_tangent_velocity(
    position: xr.DataArray,
) -> xr.DataArray:
    r"""Decompose pairwise relative motion into radial and tangential parts.

    For each ordered pair ``(i, j)``, the relative velocity is projected
    onto the radial direction (along the line connecting the two
    individuals) and the tangential direction (perpendicular to it).

    Parameters
    ----------
    position
        Position data with dimensions ``(time, space, individuals)`` or
        ``(time, space, keypoints, individuals)``.

    Returns
    -------
    xarray.DataArray
        Array with dimensions
        ``(time, individuals, individuals_other, component)`` where
        ``component`` has coordinates ``["radial", "tangential"]``:

        - ``radial``: signed projection of relative velocity onto the
          unit vector from ``i`` to ``j``. Positive means the pair
          is moving apart (increasing distance).
        - ``tangential``: magnitude of the component perpendicular to
          the radial direction (parallel motion).

        Diagonal entries (self-pairs) are ``NaN``.

    Notes
    -----
    For pair :math:`(i, j)`:

    .. math::

        \hat{r}_{ij} &=
        \frac{\mathbf{p}_j - \mathbf{p}_i}
             {|\mathbf{p}_j - \mathbf{p}_i|} \\
        \mathbf{v}_\mathrm{rel} &= \mathbf{v}_j - \mathbf{v}_i \\
        v_\mathrm{rad} &=
        \mathbf{v}_\mathrm{rel} \cdot \hat{r}_{ij} \\
        v_\mathrm{tan} &=
        \left|\mathbf{v}_\mathrm{rel}
        - v_\mathrm{rad}\,\hat{r}_{ij}\right|

    References
    ----------
    .. [1] Cheng, X. et al. (2025). Multimodal analysis of collective
       behavior. *Nature Methods*.

    Examples
    --------
    >>> atv = compute_approach_tangent_velocity(ds.position)
    >>> radial = atv.sel(component="radial")
    >>> tangential = atv.sel(component="tangential")
    """
    validate_dims_coords(
        position, {"time": [], "space": [], "individuals": []}
    )

    pos = _position_for_individuals(position)
    velocity = pos.differentiate("time")  # (time, space, individuals)

    pos_other = pos.rename({"individuals": "individuals_other"})
    vel_other = velocity.rename({"individuals": "individuals_other"})

    # Relative position: from i to j
    rel_pos = pos_other - pos  # (time, space, individuals, individuals_other)
    # Relative velocity: vel_j - vel_i
    rel_vel = vel_other - velocity  # same shape

    # Unit radial vector (from i to j)
    rel_pos_norm = compute_norm(rel_pos)  # (time, individuals, individuals_other)
    unit_radial = rel_pos / rel_pos_norm  # (time, space, individuals, individuals_other)

    # Radial component: dot(rel_vel, unit_radial) summed over space
    # Use xr.where to propagate NaN when individuals are at the same position
    v_rad = xr.where(
        rel_pos_norm > 0,
        (rel_vel * unit_radial).sum(dim="space"),
        np.nan,
    )
    # (time, individuals, individuals_other)

    # Tangential component: |rel_vel - v_rad * unit_radial|
    v_tan_vec = rel_vel - v_rad * unit_radial
    v_tan = compute_norm(v_tan_vec)  # (time, individuals, individuals_other)

    v_rad_da = v_rad.assign_coords({"component": "radial"})
    v_tan_da = v_tan.assign_coords({"component": "tangential"})
    result = xr.concat([v_rad_da, v_tan_da], dim="component")

    result = result.transpose(
        "time", "individuals", "individuals_other", "component"
    )
    result.name = "approach_tangent_velocity"
    return result
