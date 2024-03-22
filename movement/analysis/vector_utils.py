import numpy as np
import xarray as xr


def cart2pol(data: xr.DataArray) -> xr.DataArray:
    """Transform Cartesian coordinates to polar.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``space`` as a dimension with
        ``x`` and ``y`` labels.

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
        data.sel(space="y"),
        data.sel(space="x"),
    )
    # Replace space dim with space_polar
    dims = list(data.dims)
    dims[dims.index("space")] = "space_polar"
    return xr.combine_nested(
        [
            rho.assign_coords({"space_polar": "rho"}),
            theta.assign_coords({"space_polar": "theta"}),
        ],
        concat_dim="space_polar",
    ).transpose(*dims)


def pol2cart(data: xr.DataArray) -> xr.DataArray:
    """Transform polar coordinates to Cartesian.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``space_polar`` as a dimension with
        ``rho`` and ``theta`` labels.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the Cartesian coordinates
        x and y.
    """
    rho = data.sel(space_polar="rho")
    theta = data.sel(space_polar="theta")
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    # Replace space_polar dim with space
    dims = list(data.dims)
    dims[dims.index("space_polar")] = "space"
    return xr.combine_nested(
        [
            x.assign_coords({"space": "x"}),
            y.assign_coords({"space": "y"}),
        ],
        concat_dim="space",
    ).transpose(*dims)


# validate space dimension with x and y
# validate space_polar dimension with rho and theta
