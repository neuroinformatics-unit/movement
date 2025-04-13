"""Validators for data arrays."""

from collections.abc import Hashable

import xarray as xr

from movement.utils.logging import logger


def validate_dims_coords(
    data: xr.DataArray,
    required_dim_coords: dict[str, list[str] | list[Hashable]],
    exact_coords: bool = False,
) -> None:
    """Validate dimensions and coordinates in a data array.

    This function raises a ValueError if the specified dimensions and
    coordinates are not present in the input data array. By default,
    each dimension must contain *at least* the specified coordinates.
    Pass ``exact_coords=True`` to require that each dimension contains
    *exactly* the specified coordinates (and no others).

    Parameters
    ----------
    data : xarray.DataArray
        The input data array to validate.
    required_dim_coords : dict of {str: list of str | list of Hashable}
        A dictionary mapping required dimensions to a list of required
        coordinate values along each dimension.
    exact_coords : bool, optional
        If False (default), checks only that the listed coordinates
        exist in each dimension. If True, checks that each dimension
        has exactly the specified coordinates and no more.
        The exactness check is completely skipped for dimensions with
        no required coordinates.

    Examples
    --------
    Validate that a data array contains the dimension 'time'. No specific
    coordinates are required.

    >>> validate_dims_coords(data, {"time": []})

    Validate that a data array contains the dimensions 'time' and 'space',
    and that the 'space' dimension contains the coordinates 'x' and 'y'.

    >>> validate_dims_coords(data, {"time": [], "space": ["x", "y"]})

    Enforce that 'space' has *only* 'x' and 'y', and no other coordinates:

    >>> validate_dims_coords(data, {"space": ["x", "y"]}, exact_coords=True)

    Raises
    ------
    ValueError
        If the input data does not contain the required dimension(s)
        and/or the required coordinate(s).

    """
    # 1. Check that all required dimensions are present
    missing_dims = [dim for dim in required_dim_coords if dim not in data.dims]
    error_message = ""
    if missing_dims:
        error_message += (
            f"Input data must contain {missing_dims} as dimensions.\n"
        )

    # 2. For each dimension, check the presence of required coords
    for dim, coords in required_dim_coords.items():
        dim_coords_in_data = data.coords.get(dim, [])
        missing_coords = [c for c in coords if c not in dim_coords_in_data]
        if missing_coords:
            error_message += (
                f"Input data must contain {missing_coords} "
                f"in the '{dim}' coordinates.\n"
            )

        # 3. If exact_coords is True, verify no extra coords exist
        if exact_coords and coords:
            extra_coords = [c for c in dim_coords_in_data if c not in coords]
            if extra_coords:
                error_message += (
                    f"Dimension '{dim}' must only contain "
                    f"{coords} as coordinates, "
                    f"but it also has {list(extra_coords)}.\n"
                )

    if error_message:
        raise logger.error(ValueError(error_message))
