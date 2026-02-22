import numpy as np
import xarray as xr

from movement.kinematics import compute_displacement
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords
from movement.utils.logging import log_to_attrs, logger

@log_to_attrs
def compute_turning_angles(
    data: xr.DataArray,
) -> xr.DataArray:
    """Compute turning angles between consecutive steps."""
    validate_dims_coords(data, {"time": [], "space": []})
    
    if data.sizes.get("space") != 2:
        raise logger.error(ValueError("Turning angles currently only support 2D spatial data."))

    disp = compute_displacement(data)
    headings = np.arctan2(disp.sel(space="y"), disp.sel(space="x"))
    turning = headings - headings.shift(time=1)
    turning = ((turning + np.pi) % (2 * np.pi)) - np.pi
    
    step_lengths = compute_norm(disp)
    invalid_steps = (step_lengths == 0) | (step_lengths.shift(time=1) == 0)
    turning = xr.where(invalid_steps, np.nan, turning)
    
    return turning
