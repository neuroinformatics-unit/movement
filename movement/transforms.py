"""Transform and add unit attributes to xarray.DataArray datasets."""

import numpy as np
import xarray as xr


def scale(
    data: xr.DataArray,
    factor: float | np.ndarray[float] = 1.0,
    unit: str | None = None,
) -> xr.DataArray:
    """Scale data by a given factor with an optional unit.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be scaled.
    factor : float or np.ndarray of floats
        The scaling factor to apply to the data. If factor is a scalar, all
        dimensions of the data array are scaled by the same factor. If factor
        is a list or an 1D array, the length of the array must match the length
        of one of the data array's dimensions. The factor is broadcast
        along the first matching dimension.
    unit : str or None
        The unit of the scaled data stored as a property in
        xarray.DataArray.attrs['unit']. In case of the default (``None``) the
        ``unit`` attribute is dropped.

    Returns
    -------
    xarray.DataArray
        The scaled data array.

    Notes
    -----
    When scale is used multiple times on the same xarray.DataArray,
    xarray.DataArray.attrs["unit"] is overwritten each time or is dropped if
    ``None`` is passed by default or explicitly.

    When the factor is a scalar (a single number), the scaling factor is
    applied to all dimensions, while if the factor is a list or array, the
    factor is broadcasted along the first matching dimension.

    """
    if not np.isscalar(factor):
        factor = np.array(factor).squeeze()
        if factor.ndim != 1:
            raise ValueError(
                f"Factor must be a scalar or a 1D array, got {factor.ndim}D"
            )
        elif factor.shape[0] not in data.shape:
            raise ValueError(
                f"Factor shape {factor.shape} does not match "
                f"the length of any data axes: {data.shape}"
            )
        else:
            matching_dims = np.array(data.shape) == factor.shape[0]
            first_matching_dim = np.argmax(matching_dims).item()
            factor_dims = [1] * data.ndim
            factor_dims[first_matching_dim] = factor.shape[0]
            factor = factor.reshape(factor_dims)
    scaled_data = data * factor

    if unit is not None:
        scaled_data.attrs["unit"] = unit
    elif unit is None:
        scaled_data.attrs.pop("unit", None)
    return scaled_data
