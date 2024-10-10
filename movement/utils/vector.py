"""Utility functions for vector operations."""

import numpy as np
import xarray as xr

from movement.utils.logging import log_error
from movement.validators.arrays import validate_dims_coords


def compute_norm(data: xr.DataArray) -> xr.DataArray:
    """Compute the norm of the vectors along the spatial dimension.

    The norm of a vector is its magnitude, also called Euclidean norm, 2-norm
    or Euclidean length. Note that if the input data is expressed in polar
    coordinates, the magnitude of a vector is the same as its radial coordinate
    ``rho``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array containing either ``space`` or ``space_pol``
        as a dimension.

    Returns
    -------
    xarray.DataArray
         A data array holding the norm of the input vectors.
         Note that this output array has no spatial dimension but preserves
         all other dimensions of the input data array (see Notes).

    Notes
    -----
    If the input data array is a ``position`` array, this function will compute
    the magnitude of the position vectors, for every individual and keypoint,
    at every timestep. If the input data array is a ``shape`` array of a
    bounding boxes dataset, it will compute the magnitude of the shape
    vectors (i.e., the diagonal of the bounding box),
    for every individual and at every timestep.


    """
    if "space" in data.dims:
        validate_dims_coords(data, {"space": ["x", "y"]})
        return xr.apply_ufunc(
            np.linalg.norm,
            data,
            input_core_dims=[["space"]],
            kwargs={"axis": -1},
        )
    elif "space_pol" in data.dims:
        validate_dims_coords(data, {"space_pol": ["rho", "phi"]})
        return data.sel(space_pol="rho", drop=True)
    else:
        _raise_error_for_missing_spatial_dim()


def convert_to_unit(data: xr.DataArray) -> xr.DataArray:
    """Convert the vectors along the spatial dimension into unit vectors.

    A unit vector is a vector pointing in the same direction as the original
    vector but with norm = 1.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array containing either ``space`` or ``space_pol``
        as a dimension.

    Returns
    -------
    xarray.DataArray
        A data array holding the unit vectors of the input data array
        (all input dimensions are preserved).

    Notes
    -----
    Note that the unit vector for the null vector is undefined, since the null
    vector has 0 norm and no direction associated with it.

    """
    if "space" in data.dims:
        validate_dims_coords(data, {"space": ["x", "y"]})
        return data / compute_norm(data)
    elif "space_pol" in data.dims:
        validate_dims_coords(data, {"space_pol": ["rho", "phi"]})
        # Set both rho and phi values to NaN at null vectors (where rho = 0)
        new_data = xr.where(data.sel(space_pol="rho") == 0, np.nan, data)
        # Set the rho values to 1 for non-null vectors (phi is preserved)
        new_data.loc[{"space_pol": "rho"}] = xr.where(
            new_data.sel(space_pol="rho").isnull(), np.nan, 1
        )
        return new_data
    else:
        _raise_error_for_missing_spatial_dim()


def cart2pol(data: xr.DataArray) -> xr.DataArray:
    """Transform Cartesian coordinates to polar.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``space`` as a dimension,
        with ``x`` and ``y`` in the dimension coordinate.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the polar coordinates
        stored in the ``space_pol`` dimension, with ``rho``
        and ``phi`` in the dimension coordinate. The angles
        ``phi`` returned are in radians, in the range ``[-pi, pi]``.

    """
    validate_dims_coords(data, {"space": ["x", "y"]})
    rho = compute_norm(data)
    phi = xr.apply_ufunc(
        np.arctan2,
        data.sel(space="y"),
        data.sel(space="x"),
    )
    # Replace space dim with space_pol
    dims = list(data.dims)
    dims[dims.index("space")] = "space_pol"
    return xr.combine_nested(
        [
            rho.assign_coords({"space_pol": "rho"}),
            phi.assign_coords({"space_pol": "phi"}),
        ],
        concat_dim="space_pol",
    ).transpose(*dims)


def pol2cart(data: xr.DataArray) -> xr.DataArray:
    """Transform polar coordinates to Cartesian.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``space_pol`` as a dimension,
        with ``rho`` and ``phi`` in the dimension coordinate.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the Cartesian coordinates
        stored in the ``space`` dimension, with ``x`` and ``y``
        in the dimension coordinate.

    """
    validate_dims_coords(data, {"space_pol": ["rho", "phi"]})
    rho = data.sel(space_pol="rho")
    phi = data.sel(space_pol="phi")
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    # Replace space_pol dim with space
    dims = list(data.dims)
    dims[dims.index("space_pol")] = "space"
    return xr.combine_nested(
        [
            x.assign_coords({"space": "x"}),
            y.assign_coords({"space": "y"}),
        ],
        concat_dim="space",
    ).transpose(*dims)


def signed_angle_between_2d_vectors(
    test_vector: xr.DataArray, reference_vector: xr.DataArray | np.ndarray
) -> xr.DataArray:
    """Compute the signed angle between two 2-D vectors.

    Parameters
    ----------
    test_vector : xarray.DataArray
        An array of position vectors containing the ``space``
        dimension with only ``"x"`` and ``"y"`` coordinates.
    reference_vector : xarray.DataArray | numpy.ndarray
        A 2D vector (or array of 2D vectors) against which to
        compare ``test_vector``. May either be an xarray
        DataArray containing the ``space``  dimension or a numpy
        array containing one or more 2D vectors. (See Notes)

    Returns
    -------
    xarray.DataArray :
        An xarray DataArray containing signed angle between
        ``test_vector`` and ``reference_vector`` for every
        time-point. Matches the dimensions of ``test_vector``,
        but without the ``space`` dimension.

    Notes
    -----
    If passed as an xarray DataArray, the reference vector must
    have the spatial coordinates ``x`` and ``y`` only, and must
    have a ``time`` dimension matching that of the test vector.

    If passed as a numpy array, the reference vector must have
    one of three shapes:
        1. ``(2,)`` - Where dimension ``0`` contains spatial
        coordinates (x,y), and no time dimension is specified.
        2. ``(1,2)`` - Where dimension ``0`` corresponds to a
        single time-point and dimension ``1`` contains spatial
        coordinates (x,y).
        3. ``(n,2)`` - Where dimension ``0`` corresponds to
        time and dimension ``1`` contains spatial coordinates
        (x,y), and where ``n == len(test_vector.time)``.

    Reference vectors containing more dimensions, or with shapes
    otherwise different from those defined above are considered
    invalid.

    """
    if isinstance(reference_vector, np.ndarray) and reference_vector.shape == (
        2,
    ):
        reference_vector = reference_vector.reshape(1, 2)

    validate_dims_coords(test_vector, {"space": ["x", "y"]})
    _validate_reference_vector(reference_vector, test_vector)

    test_unit = convert_to_unit(test_vector)
    test_x = test_unit.sel(space="x")
    test_y = test_unit.sel(space="y")

    if isinstance(reference_vector, xr.DataArray):
        ref_unit = convert_to_unit(reference_vector)
        ref_x = ref_unit.sel(space="x")
        ref_y = ref_unit.sel(space="y")
    else:
        ref_unit = reference_vector / np.linalg.norm(reference_vector)
        ref_x = np.take(ref_unit, 0, axis=-1).reshape(-1, 1)
        ref_y = np.take(ref_unit, 1, axis=-1).reshape(-1, 1)

    signed_angles = np.arctan2(
        test_y * ref_x - test_x * ref_y,
        test_x * ref_x + test_y * ref_y,
    )

    return signed_angles


def _raise_error_for_missing_spatial_dim() -> None:
    raise log_error(
        ValueError,
        "Input data array must contain either 'space' or 'space_pol' "
        "as dimensions.",
    )


def _validate_reference_vector(
    reference_vector: xr.DataArray | np.ndarray, test_vector: xr.DataArray
):
    """Validate the reference vector has the correct type and dimensions.

    Parameters
    ----------
    reference_vector : xarray.DataArray | numpy.ndarray
        The reference vector array to validate.
    test_vector : xarray.DataArray
        The input data against which to validate the
        reference vector.

    Returns
    -------
    TypeError
        If reference_vector is not an xarray DataArray or
        a numpy array
    ValueError
        If reference_vector does not have the correct dimensions

    """
    # Validate reference vector type
    if not isinstance(reference_vector, (xr.DataArray | np.ndarray)):
        raise log_error(
            TypeError,
            f"Reference vector must be an xarray.DataArray or a np.ndarray, "
            f"but got {type(reference_vector)}.",
        )
    if isinstance(reference_vector, xr.DataArray):
        validate_dims_coords(
            reference_vector,
            {
                "space": ["x", "y"],
            },
        )
        # Check reference_vector is 2D
        if len(reference_vector.space) > 2:
            raise log_error(
                ValueError,
                "Reference vector may not have more than 2 spatial "
                "coordinates.",
            )
        # Check reference vector has valid time dimension
        if "time" in reference_vector.dims and not len(
            reference_vector.time
        ) == len(test_vector.time):
            raise log_error(
                ValueError,
                "Input data and reference vector must have matching time "
                "dimensions.",
            )
        if any(dim not in ["time", "space"] for dim in reference_vector.dims):
            raise log_error(
                ValueError, "Reference vector contains invalid dimensions."
            )
    else:
        if not (reference_vector.dtype == int) or (
            reference_vector.dtype == float
        ):
            raise log_error(
                ValueError,
                "Reference vector may only contain values of type ``int``"
                "or ``float``.",
            )
        if not (
            reference_vector.shape[0] == 1
            or reference_vector.shape[0] == len(test_vector.time)
        ):  # Validate time dim
            raise log_error(
                ValueError,
                "Dimension ``0`` of the reference vector must have length "
                "``1`` or be equal in length to the ``time`` dimension of the "
                "test vector.",
            )
        if not reference_vector.shape[-1] == 2:  # Validate space dimension
            raise log_error(
                ValueError,
                "Dimension ``-1`` of the reference_vector must correspond to "
                "coordinates in 2-D space, and may therefore only have size "
                f"``2``. Instead, got size ``{reference_vector.shape[1]}``.",
            )
        if len(reference_vector.shape) > 2:
            raise log_error(
                ValueError,
                "Reference vector may not have more than 2 dimensions (time"
                "and space, respectively)",
            )
