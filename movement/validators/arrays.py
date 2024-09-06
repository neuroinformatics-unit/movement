"""Validators for data arrays."""

import xarray as xr

from movement.utils.logging import log_error


def validate_dims_coords(
    data: xr.DataArray, required_dim_coords: dict
) -> None:
    """Validate dimensions and coordinates in a data array.

    This function raises a ValueError if the specified dimensions and
    coordinates are not present in the input data array.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array to validate.
    required_dim_coords : dict
        A dictionary of required dimensions and their corresponding
        coordinate values. If you don't need to specify any
        coordinate values, you can pass an empty list.

    Examples
    --------
    Validate that a data array contains the dimension 'time'. No specific
    coordinates are required.

    >>> validate_dims_coords(data, {"time": []})

    Validate that a data array contains the dimensions 'time' and 'space',
    and that the 'space' dimension contains the coordinates 'x' and 'y'.

    >>> validate_dims_coords(data, {"time": [], "space": ["x", "y"]})

    Raises
    ------
    ValueError
        If the input data does not contain the required dimension(s)
        and/or the required coordinate(s).

    """
    missing_dims = [dim for dim in required_dim_coords if dim not in data.dims]
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
