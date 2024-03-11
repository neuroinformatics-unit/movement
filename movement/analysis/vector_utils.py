import numpy as np
import xarray as xr


def cart2polar(data: xr.DataArray) -> xr.DataArray:
    """Transform Cartesian coordinates to polar.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``space`` as a dimension.
    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the polar coordinates
        rho and theta.
    """
    rho = xr.apply_ufunc(
        np.linalg.norm,
        data,
        input_core_dims=[["space"]],
        kwargs={"axis": -1},
    )
    theta = xr.apply_ufunc(
        np.arctan2,
        data[..., 1],
        data[..., 0],
    )
    return xr.combine_nested(
        [
            rho.assign_coords({"space_polar": "rho"}),
            theta.assign_coords({"space_polar": "theta"}),
        ],
        concat_dim="space_polar",
    )


# validate space dimension
