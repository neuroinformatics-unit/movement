import numpy as np
import xarray as xr

from movement.logging import log_error


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
    _validate_dimension_coordinates(data, {"space": ["x", "y"]})
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
    _validate_dimension_coordinates(data, {"space_polar": ["rho", "theta"]})
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


def _validate_dimension_coordinates(
    data: xr.DataArray, required_dim_coords: dict
) -> None:
    """Validate the input data contains the required dimensions and
    coordinate values.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to validate.
    required_dim_coords : dict
        A dictionary of required dimensions and their corresponding
        coordinate values.

    Raises
    ------
    ValueError
        If the input data does not contain the required dimension(s)
        and/or the required coordinate(s).
    """
    missing_dims = [
        dim for dim in required_dim_coords.keys() if dim not in data.dims
    ]
    error_message = ""
    if missing_dims:
        error_message += (
            f"Input data must contain {missing_dims} as dimensions.\n"
        )
    missing_coords = []
    for dim, coords in required_dim_coords.items():
        missing_coords = [
            coord for coord in coords if coord not in data.coords.get(dim, [])
        ]
        if missing_coords:
            error_message += (
                "Input data must contain "
                f"{missing_coords} in the '{dim}' coordinates."
            )
    if error_message:
        raise log_error(ValueError, error_message)
