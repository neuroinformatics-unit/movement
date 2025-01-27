"""Transform and add unit attributes to xarray.DataArray datasets."""

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike


def scale(
    data: xr.DataArray,
    factor: ArrayLike = 1.0,
    space_unit: str | None = None,
) -> xr.DataArray:
    """Scale data by a given factor with an optional unit.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be scaled.
    factor : ArrayLike
        The scaling factor to apply to the data. Any object that can be
        converted to a 1D numpy array is valid (e.g. a single float or a list
        of floats). If factor is a single float, the data array is uniformly
        scaled by the same factor. If factor contains multiple floats, the
        length of the resulting array must match the length of data array's
        unit dimension along which it will be broadcasted.
    space_unit : str or None
        The unit of the scaled data stored as a property in
        xarray.DataArray.attrs['space_unit']. In case of the default (``None``)
        the ``space_unit`` attribute is dropped.

    Returns
    -------
    xarray.DataArray
        The scaled data array.

    Notes
    -----
    When scale is used multiple times on the same xarray.DataArray,
    xarray.DataArray.attrs["space_unit"] is overwritten each time or is dropped
    if ``None`` is passed by default or explicitly.

    When the factor is a scalar (a single number), the scaling factor is
    applied to all dimensions, while if the factor is a list or array, the
    factor is broadcasted along the first matching dimension.

    """
    if not np.isscalar(factor):
        factor = np.array(factor).squeeze()
        if factor.ndim != 1:
            raise ValueError(
                "Factor must be an object that can be converted to a 1D numpy"
                f" array, got {factor.ndim}D"
            )
        elif factor.shape[0] != data.space.values.shape[0]:
            raise ValueError(
                f"Factor length {factor.shape[0]} does not match the length "
                f"of the space dimension {data.space.values.shape[0]}"
            )
        else:
            factor_dims = [1] * data.ndim  # 1s array matching data dimensions
            factor_dims[data.get_axis_num("space")] = factor.shape[0]
            factor = factor.reshape(factor_dims)
    scaled_data = data * factor

    if space_unit is not None:
        scaled_data.attrs["space_unit"] = space_unit
    elif space_unit is None:
        scaled_data.attrs.pop("space_unit", None)
    return scaled_data
