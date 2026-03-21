import warnings
from typing import Literal
import xarray as xr
from movement.utils.logging import logger
from movement.utils.reports import report_nan_values
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords
def _smooth_data(data: xr.DataArray, window: int) -> xr.DataArray:
    """Apply simple moving average smoothing along time dimension."""
    if window <= 1:
        return data
    return data.rolling(time=window, center=True, min_periods=1).mean()
def compute_time_derivative(
    data: xr.DataArray,
    order: int,
    smooth: bool = False,
    window: int = 3,
) -> xr.DataArray:
    """Compute the time-derivative of an array using numerical differentiation.
    Optionally applies smoothing before differentiation to reduce noise.
    Parameters
    ----------
    data
        The input data containing ``time`` as a required dimension.
    order
        Order of derivative (1 = velocity, 2 = acceleration).
    smooth
        Whether to apply smoothing before differentiation.
    window
        Rolling window size for smoothing.
    Returns
    -------
    xarray.DataArray
    """
    if not isinstance(order, int):
        raise logger.error(
            TypeError(f"Order must be an integer, but got {type(order)}.")
        )
    if order <= 0:
        raise logger.error(ValueError("Order must be a positive integer."))
    validate_dims_coords(data, {"time": []})
    if smooth:
        data = _smooth_data(data, window)
    result = data
    for _ in range(order):
        result = result.differentiate("time")
    return result
def compute_displacement(data: xr.DataArray) -> xr.DataArray:
    warnings.warn(
        "compute_displacement is deprecated. Use forward/backward displacement.",
        DeprecationWarning,
        stacklevel=2,
    )
    validate_dims_coords(data, {"time": [], "space": []})
    result = data.diff(dim="time")
    result = result.reindex_like(data, fill_value=0)
    result.name = "displacement"
    return result
def _compute_forward_displacement(data: xr.DataArray) -> xr.DataArray:
    validate_dims_coords(data, {"time": [], "space": []})
    result = data.diff(dim="time", label="lower")
    result = result.reindex_like(data, fill_value=0)
    return result
def compute_forward_displacement(data: xr.DataArray) -> xr.DataArray:
    result = _compute_forward_displacement(data)
    result.name = "forward_displacement"
    return result
def compute_backward_displacement(data: xr.DataArray) -> xr.DataArray:
    fwd = _compute_forward_displacement(data)
    result = -fwd.roll(time=1)
    result.name = "backward_displacement"
    return result
def compute_velocity(
    data: xr.DataArray,
    smooth: bool = False,
    window: int = 3,
) -> xr.DataArray:
    validate_dims_coords(data, {"space": []})
    result = compute_time_derivative(
        data, order=1, smooth=smooth, window=window
    )
    result.name = "velocity"
    return result
def compute_acceleration(
    data: xr.DataArray,
    smooth: bool = False,
    window: int = 3,
) -> xr.DataArray:
    validate_dims_coords(data, {"space": []})
    result = compute_time_derivative(
        data, order=2, smooth=smooth, window=window
    )
    result.name = "acceleration"
    return result
def compute_speed(data: xr.DataArray) -> xr.DataArray:
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
    validate_dims_coords(data, {"time": [], "space": []})
    data = data.sel(time=slice(start, stop))
    if data.sizes["time"] < 2:
        raise logger.error(ValueError("At least 2 time points required."))
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
        raise logger.error(ValueError("Invalid nan_policy."))
    result.name = "path_length"
    return result
def _warn_about_nan_proportion(
    data: xr.DataArray, nan_warn_threshold: float
) -> None:
    if not 0 <= nan_warn_threshold <= 1:
        raise logger.error(ValueError("Threshold must be between 0 and 1."))
    n_nans = data.isnull().any(dim="space").sum(dim="time")
    warn_data = data.where(
        n_nans >= data.sizes["time"] * nan_warn_threshold, drop=True
    )
    if warn_data.size > 0:
        warnings.warn(
            f"High NaN proportion detected:\n{report_nan_values(warn_data)}",
            UserWarning,
        )
def _compute_scaled_path_length(data: xr.DataArray) -> xr.DataArray:
    disp = compute_backward_displacement(data).isel(time=slice(1, None))
    valid = (~disp.isnull()).all(dim="space").sum(dim="time")
    prop = valid / (data.sizes["time"] - 1)
    return compute_norm(disp).sum(dim="time") / prop
