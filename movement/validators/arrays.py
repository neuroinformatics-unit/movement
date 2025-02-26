"""Validators for data arrays."""

from collections.abc import Hashable

import numpy as np
import xarray as xr

from movement.utils.logging import log_error


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
        raise log_error(ValueError, error_message)


def validate_reference_vector(
    reference_vector: xr.DataArray | np.ndarray,
    test_vector: xr.DataArray,
) -> xr.DataArray:
    """Validate a reference vector being used in a calculation.

    Reference vectors must contain the same number of time points as the
    ``test_vector``, in order to be used in computations with them.
    Vector-like objects that are passed in are converted into ``xr.DataArray``s
    and inherit the ``test_vector``s ``"time"`` axis.

    Parameters
    ----------
    reference_vector : xarray.DataArray | numpy.ndarray
        The reference vector to validate, and convert if necessary.
    test_vector : xarray.DataArray
        The test vector to compare against. The reference vector must have
        the same number of time points as the test vector.

    Returns
    -------
    xarray.DataArray
        A validated xarray DataArray.

    Raises
    ------
    ValueError
        If shape or dimensions do not match the expected form.
    TypeError
        If reference_vector is neither a NumPy array nor an xarray DataArray.

    """
    if isinstance(reference_vector, np.ndarray):
        # Check shape: must be 1D or 2D
        if reference_vector.ndim > 2:
            raise log_error(
                ValueError,
                "Reference vector must be 1D or 2D, but got "
                f"{reference_vector.ndim}D array.",
            )
        # Reshape 1D -> (1, -1) so axis 0 can be 'time'
        if reference_vector.ndim == 1:
            reference_vector = reference_vector.reshape(1, -1)

        # If multiple time points, must match test_vector time length
        if (
            reference_vector.shape[0] > 1
            and reference_vector.shape[0] != test_vector.sizes["time"]
        ):
            raise log_error(
                ValueError,
                "Reference vector must have the same number of time "
                "points as the test vector.",
            )

        # Decide whether we have (time, space) or just (space)
        if reference_vector.shape[0] == 1:
            coords = {"space": ["x", "y"]}
            reference_vector = reference_vector.squeeze()
        else:
            coords = {
                "time": test_vector["time"],
                "space": ["x", "y"],
            }

        return xr.DataArray(
            reference_vector, dims=list(coords.keys()), coords=coords
        )

    elif isinstance(reference_vector, xr.DataArray):
        # Must contain exactly 'x' and 'y' in the space dimension
        validate_dims_coords(
            reference_vector, {"space": ["x", "y"]}, exact_coords=True
        )

        # If it has a time dimension, time size must match
        if (
            "time" in reference_vector.dims
            and reference_vector.sizes["time"] != test_vector.sizes["time"]
        ):
            raise log_error(
                ValueError,
                "Reference vector must have the same number of time "
                "points as the test vector.",
            )

        # Only 'time' and 'space' are allowed
        if any(d not in {"time", "space"} for d in reference_vector.dims):
            raise log_error(
                ValueError,
                "Only 'time' and 'space' dimensions "
                "are allowed in reference_vector.",
            )
        return reference_vector

    # If it's neither a DataArray nor a NumPy array
    raise log_error(
        TypeError,
        "Reference vector must be an xarray.DataArray or np.ndarray, "
        f"but got {type(reference_vector)}.",
    )
