"""Straightness index implementation for trajectory_complexity.py."""

import numpy as np
import xarray as xr

from movement.kinematics import compute_path_length
from movement.utils.logging import log_to_attrs, logger
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


@log_to_attrs
def compute_straightness_index(
    data: xr.DataArray,
    window_size: int | None = None,
) -> xr.DataArray:
    """Compute the straightness index of a trajectory.

    The straightness index is defined as the ratio of the Euclidean distance
    between the start and end points of a trajectory to the total path
    length (D / L). Values range from 0 to 1, where 1 indicates a
    perfectly straight path and 0 indicates the animal returned to its
    starting point.

    Parameters
    ----------
    data : xr.DataArray
        The input position data. Must contain ``time`` and ``space``
        dimensions.
    window_size : int, optional
        If provided, compute the straightness index over a rolling window
        of this many time steps. The result is a time series of local
        straightness values rather than a single global value. If ``None``
        (default), the global straightness index is computed over the
        entire trajectory.

    Returns
    -------
    xr.DataArray
        If ``window_size`` is ``None``: a DataArray with the ``time``
        and ``space`` dimensions reduced, containing one straightness
        value per individual/keypoint combination.

        If ``window_size`` is provided: a DataArray with the same shape
        as the input ``data`` (minus ``space``), containing the rolling
        straightness index at each time step. The first
        ``window_size - 1`` values are ``NaN`` as the window is not yet
        full.

    Raises
    ------
    ValueError
        If ``data`` does not contain ``time`` and ``space`` dimensions,
        or if ``window_size`` is not a positive integer.

    Notes
    -----
    **Global straightness** (``window_size=None``):
    Computes S = D / L where D is the Euclidean distance between the
    first and last positions, and L is the total path length. Returns
    a single scalar per individual/keypoint.

    **Rolling straightness** (``window_size=W``):
    Computes S(t) = D(t) / L(t) where:

    - D(t) = ||position(t) - position(t - W)|| (rolling displacement)
    - L(t) = sum of step lengths over the window [t-W, t]

    This is computed using vectorized xarray operations with no Python
    loops, and is therefore as efficient for 100 individuals as for 1.
    The output is aligned to the original time axis with the first
    ``window_size - 1`` values set to ``NaN``.

    **NaN behaviour:** NaN positions propagate to NaN straightness values.
    For best results, apply :func:`movement.filtering.interpolate_over_time`
    before computing the straightness index.

    **Path length of zero:** If an animal is stationary, the path length
    is zero and the straightness index is undefined (0 / 0). These cases
    are set to ``NaN`` rather than raising a ``ZeroDivisionError``.

    See Also
    --------
    movement.kinematics.compute_path_length : Total path length.
    movement.kinematics.compute_turning_angles : Turning angles along
        the trajectory.

    Examples
    --------
    Compute global straightness from a ``movement`` poses dataset:

    >>> from movement import sample_data
    >>> from movement.kinematics import compute_straightness_index
    >>> ds = sample_data.fetch_dataset("DLC_single-mouse_EPM.predictions.h5")
    >>> si = compute_straightness_index(ds.position)

    Compute rolling straightness over 50-frame windows:

    >>> rolling_si = compute_straightness_index(ds.position, window_size=50)

    """
    validate_dims_coords(data, {"time": [], "space": []})

    if window_size is not None and (
        not isinstance(window_size, int) or window_size < 1
    ):
        raise logger.error(
            ValueError(
                f"window_size must be a positive integer, got {window_size}."
            )
        )

    if window_size is None:
        # ── Global straightness ──────────────────────────────────────────
        start_point = data.isel(time=0)
        end_point = data.isel(time=-1)
        displacement = compute_norm(end_point - start_point)
        path_length = compute_path_length(data)

    else:
        # ── Rolling straightness ─────────────────────────────────────────
        # D(t): displacement from W steps ago to now — fully vectorized
        displacement = compute_norm(data - data.shift(time=window_size))

        # L(t): rolling sum of step lengths over the window
        # compute_path_length uses diff internally; we need step lengths
        step_lengths = compute_norm(data.diff(dim="time"))
        # Align step_lengths to the same time axis as data (diff drops t=0)
        step_lengths = step_lengths.reindex(
            time=data.coords["time"], fill_value=np.nan
        )
        path_length = step_lengths.rolling(
            time=window_size, min_periods=window_size
        ).sum()

    # Avoid 0/0 → NaN rather than RuntimeWarning
    result = xr.where(path_length > 0, displacement / path_length, np.nan)

    # Drop space dimension if it survived (global case)
    if "space" in result.dims:
        result = result.isel(space=0, drop=True)

    result.name = "straightness_index"
    result.attrs["units"] = "dimensionless"
    result.attrs["long_name"] = "Straightness Index (D/L)"

    return result
