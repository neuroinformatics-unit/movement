"""Module for computing trajectory complexity metrics."""

import numpy as np
import xarray as xr

from movement.kinematics import compute_displacement
from movement.utils.logging import log_to_attrs, logger
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


@log_to_attrs
def compute_turning_angles(
    data: xr.DataArray,
    in_degrees: bool = False,
) -> xr.DataArray:
    """Compute the turning angles between consecutive steps in a trajectory.

    A turning angle is the signed angular change in heading direction between
    two consecutive displacement vectors. Positive values indicate a left
    (counter-clockwise) turn; negative values indicate a right (clockwise)
    turn.

    Parameters
    ----------
    data : xr.DataArray
        The input position data. Must contain ``time`` and ``space``
        dimensions. The ``space`` dimension must contain exactly the
        coordinates ``["x", "y"]`` (2D spatial data only).
    in_degrees : bool, optional
        If ``True``, return turning angles in degrees. Default is
        ``False`` (radians).

    Returns
    -------
    xr.DataArray
        Turning angles with the same shape as the input ``data``, but
        with the ``space`` dimension dropped. Values are wrapped to
        the interval ``(-π, π]`` (or ``(-180, 180]`` if
        ``in_degrees=True``). The first two time steps are always
        ``NaN``, as at least two steps are needed to compute one
        turning angle.

    Raises
    ------
    ValueError
        If ``data`` does not contain ``time`` and ``space`` dimensions,
        or if ``space`` does not contain exactly 2 coordinates.

    Notes
    -----
    **Time dimension length:** This function uses a ``shift`` operation
    to preserve the original ``time`` dimension length. The first two
    time steps are always ``NaN``: the first because no previous step
    exists, and the second because a turning angle requires two steps
    (three positions).

    **Zero-length steps:** When an animal is stationary, the displacement
    is the zero vector and the heading is mathematically undefined.
    ``np.arctan2(0, 0)`` returns ``0.0`` in NumPy without raising an
    error, which would silently produce meaningless turning angles.
    Any turning angle involving a zero-length incoming or outgoing step
    is explicitly set to ``NaN``.

    **NaN propagation:** NaN positions in the input propagate to NaN
    turning angles. A single missing position affects up to two turning
    angles (the incoming and outgoing steps). Use
    :func:`movement.filtering.interpolate_over_time` to fill positional
    gaps before computing turning angles if continuity is important.

    **Angle wrapping:** Raw heading differences are wrapped to
    ``(-π, π]`` using the modulo identity
    ``((Δθ + π) % (2π)) - π``, ensuring for example that a 350°
    raw change is reported as -10°.

    See Also
    --------
    movement.kinematics.compute_forward_displacement : The displacement
        vectors used to compute headings.
    movement.kinematics.compute_path_length : Total path length.
    movement.kinematics.straightness_index : Trajectory straightness
        computed from path and displacement.

    Examples
    --------
    Compute turning angles from a ``movement`` poses dataset:

    >>> from movement import sample_data
    >>> from movement.kinematics import compute_turning_angles
    >>> ds = sample_data.fetch_dataset("DLC_single-wasp.predictions.h5")
    >>> angles = compute_turning_angles(ds.position)

    Compute in degrees and check the distribution:

    >>> import numpy as np
    >>> angles_deg = compute_turning_angles(ds.position, in_degrees=True)
    >>> mean_turn = np.nanmean(np.abs(angles_deg.values))

    """
    validate_dims_coords(data, {"time": [], "space": []})

    if data.sizes.get("space") != 2:
        raise logger.error(
            ValueError(
                "compute_turning_angles requires exactly 2 spatial "
                f"coordinates, got {data.sizes.get('space')}. "
                "Only 2D data is supported."
            )
        )

    # Displacement vectors: shape same as data, space dimension preserved
    disp = compute_displacement(data)

    # Absolute heading per step: arctan2(dy, dx) -> shape drops "space"
    headings = np.arctan2(disp.sel(space="y"), disp.sel(space="x"))

    # Turning angle = heading difference between consecutive steps
    # shift(time=1) aligns each heading with the previous one
    turning = headings - headings.shift(time=1)  # type: ignore[union-attr]

    # Wrap to (-π, π] using modulo identity
    turning = ((turning + np.pi) % (2 * np.pi)) - np.pi

    # Mask turning angles involving zero-length steps
    # (stationary animal -> heading undefined -> arctan2(0,0) = 0 silently)
    step_lengths = compute_norm(disp)
    invalid_steps = (step_lengths == 0) | (step_lengths.shift(time=1) == 0)
    turning = xr.where(invalid_steps, np.nan, turning)

    turning.name = "turning_angle"
    turning.attrs["units"] = "radians"

    if in_degrees:
        turning = np.degrees(turning)
        turning.attrs["units"] = "degrees"

    return turning
