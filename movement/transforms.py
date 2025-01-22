"""Transforms module."""

import numpy as np
import xarray as xr


def scale(
    data_array: xr.DataArray,
    factor: float | np.ndarray[float] = 1.0,
    unit: str | None = None,
) -> xr.DataArray:
    """Scale data by a given factor with an optional unit."""
    if not np.isscalar(factor):
        factor = np.array(factor).squeeze()
        if factor.ndim != 1:
            raise ValueError(
                f"Factor must be a scalar or a 1D array, got {factor.ndim}D"
            )
        elif factor.shape[0] not in data_array.shape:
            raise ValueError(
                f"Factor shape {factor.shape} does not match "
                f"the length of any data axes: {data_array.shape}"
            )
        else:
            # To figure out which dimension to broadcast along.
            # Find dimensions with as many values as we have factors.
            matching_dims = np.array(data_array.shape) == factor.shape[0]
            # Find first dimension that matches.
            first_matching_dim = np.argmax(matching_dims).item()
            # Reshape factor to broadcast along the matching dimension.
            factor_dims = [1] * data_array.ndim
            factor_dims[first_matching_dim] = factor.shape[0]
            # Reshape factor for broadcasting.
            factor = factor.reshape(factor_dims)
    scaled_data_array = data_array * factor

    if unit is not None:
        scaled_data_array.attrs["unit"] = unit
    elif unit is None:
        scaled_data_array.attrs.pop("unit", None)
    return scaled_data_array
