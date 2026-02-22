"""Module for computing trajectory complexity metrics."""

import numpy as np
import xarray as xr

from movement.kinematics import compute_backward_displacement
from movement.utils.logging import log_to_attrs, logger
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


@log_to_attrs
def compute_turning_angles(
    data: xr.DataArray,
) -> xr.DataArray:
    """Compute the turning angles between consecutive steps in a trajectory.

    A turning angle is the signed angular change in heading direction between
    two consecutive displacement vectors. Positive values indicate a left
    (counter-clockwise) turn; negative values indicate a right (clockwise)
    turn.

    Parameters
    ----------
    data : xr.DataArray
        The input position data. Must contain `time` and `space`
        dimensions. The `space` dimension must contain exactly the
        coordinates `["x", "y"]` (2D spatial data only).

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the turning angles in radians,
        wrapped between -\u03c0 and \u03c0. The returned array preserves
        the original `time` dimension of the input `data`, but the
        `space` dimension is dropped (as angles are scalar per time step).

    Raises
    ------
    ValueError
        If the `space` dimension does not contain exactly 2 coordinates.

    Notes
    -----
    To preserve the original length of the `time` dimension, this function
    uses a shift operation rather than a difference operation. Consequently,
    the first two time steps of the resulting array will naturally be `NaN`,
    as it requires three positions (two steps) to form a single turning angle.

    If a step has a length of zero (i.e., the tracked subject was stationary),
    the heading is mathematically undefined. To prevent artificial 0.0 radian
    turning angles caused by `arctan2(0, 0)`, any turning angle involving an
    incoming or outgoing zero-length step is explicitly masked with `NaN`.

    """
    validate_dims_coords(data, {"time": [], "space": []})

    if data.sizes.get("space") != 2:
        raise logger.error(
            ValueError(
                "Turning angles currently only support 2D spatial data."
            )
        )

    disp = compute_backward_displacement(data)
    headings = np.arctan2(disp.sel(space="y"), disp.sel(space="x"))
    turning = headings - headings.shift(time=1)  # type: ignore[union-attr]
    turning = ((turning + np.pi) % (2 * np.pi)) - np.pi

    step_lengths = compute_norm(disp)
    invalid_steps = (step_lengths == 0) | (step_lengths.shift(time=1) == 0)
    turning = xr.where(invalid_steps, np.nan, turning)

    return turning
