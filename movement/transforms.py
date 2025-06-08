"""Transform and add unit attributes to xarray.DataArray datasets."""

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.utils.logging import log_to_attrs
from movement.validators.arrays import validate_dims_coords


@log_to_attrs
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
        ``xarray.DataArray.attrs['space_unit']``. In case of the default
        (``None``) the ``space_unit`` attribute is dropped.

    Returns
    -------
    xarray.DataArray
        The scaled data array.

    Notes
    -----
    This function makes two changes to the resulting data array's attributes
    (``xarray.DataArray.attrs``) each time it is called:

    - It sets the ``space_unit`` attribute to the value of the parameter
      with the same name, or removes it if ``space_unit=None``.
    - It adds a new entry to the ``log`` attribute of the data array, which
      contains a record of the operations performed, including the
      parameters used, as well as the datetime of the function call.

    Examples
    --------
    Let's imagine a camera viewing a 2D plane from the top, with an
    estimated resolution of 10 pixels per cm. We can scale down
    position data by a factor of 1/10 to express it in cm units.

    >>> from movement.transforms import scale
    >>> ds["position"] = scale(ds["position"], factor=1 / 10, space_unit="cm")
    >>> print(ds["position"].space_unit)
    cm
    >>> print(ds["position"].log)
    [
        {
            "operation": "scale",
            "datetime": "2025-06-05 15:08:16.919947",
            "factor": "0.1",
            "space_unit": "'cm'"
        }
    ]

    Note that the attributes of the scaled data array now contain the assigned
    ``space_unit`` as well as a ``log`` entry with the arguments passed to
    the function.

    We can also scale the two spatial dimensions by different factors.

    >>> ds["position"] = scale(ds["position"], factor=[10, 20])

    The second scale operation restored the x axis to its original scale,
    and scaled up the y axis to twice its original size.
    The log will now contain two entries, but the ``space_unit`` attribute
    has been removed, as it was not provided in the second function call.

    >>> "space_unit" in ds["position"].attrs
    False

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
