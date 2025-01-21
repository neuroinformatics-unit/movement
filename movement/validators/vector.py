"""Validators for vector or vector-castable objects."""

import numpy as np
import xarray as xr

from movement.utils.logging import log_error
from movement.validators.arrays import validate_dims_coords


def validate_reference_vector(
    reference_vector: xr.DataArray | np.ndarray,
    test_vector: xr.DataArray,
) -> xr.DataArray:
    """Validate a reference vector being used in a calculation.

    Reference vectors must contain the same number of time points as the
    `test_vector`, in order to be used in computations with them.
    Vector-like objects that are passed in are converted into `xr.DataArray`s
    and inherit the `test_vector`s `time` axes.

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
            # reference_vector = reference_vector.squeeze()
        else:
            coords = {
                "time": test_vector.get("time", None),
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
                "Only dimensions 'time' and 'space' dimensions "
                "are allowed in reference_vector.",
            )
        return reference_vector

    # If it's neither a DataArray nor a NumPy array
    raise log_error(
        TypeError,
        "Reference vector must be an xarray.DataArray or np.ndarray, "
        f"but got {type(reference_vector)}.",
    )
