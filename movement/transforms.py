"""Transform and add unit attributes to xarray.DataArray datasets."""

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.validators.arrays import validate_dims_coords


def scale(
    data: xr.DataArray,
    factor: ArrayLike | float = 1.0,
    space_unit: str | None = None,
) -> xr.DataArray:
    """Scale data by a given factor with an optional unit.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be scaled.
    factor : ArrayLike or float
        The scaling factor to apply to the data. If factor is a scalar (a
        single float), the data array is uniformly scaled by the same factor.
        If factor is an object that can be converted to a 1D numpy array (e.g.
        a list of floats), the length of the resulting array must match the
        length of data array's space dimension along which it will be
        broadcasted.
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

    """
    if len(data.coords["space"]) == 2:
        validate_dims_coords(data, {"space": ["x", "y"]})
    else:
        validate_dims_coords(data, {"space": ["x", "y", "z"]})

    if not np.isscalar(factor):
        factor = np.array(factor).squeeze()
        if factor.ndim != 1:
            raise ValueError(
                "Factor must be an object that can be converted to a 1D numpy"
                f" array, got {factor.ndim}D"
            )
        elif factor.shape != data.space.values.shape:
            raise ValueError(
                f"Factor shape {factor.shape} does not match the shape "
                f"of the space dimension {data.space.values.shape}"
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
