"""Utility functions for vector operations."""

from typing import NoReturn

import numpy as np
import xarray as xr

from movement.utils.logging import logger
from movement.validators.arrays import validate_dims_coords


def compute_norm(data: xr.DataArray) -> xr.DataArray:
    """Compute the norm of the vectors along the spatial dimension.

    The norm of a vector is its magnitude, also called Euclidean norm, 2-norm
    or Euclidean length. Note that if the input data is expressed in polar
    coordinates, the magnitude of a vector is the same as its radial coordinate
    ``rho``.

    Parameters
    ----------
    data
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
        # Allow both 2D and 3D 
        if len(data.coords["space"]) == 2:
            validate_dims_coords(data, {"space": ["x", "y"]})
        elif len(data.coords["space"]) == 3:
            validate_dims_coords(data, {"space": ["x", "y", "z"]})
        else:
            _raise_error_for_invalid_spatial_dim_length("space", 2, 3)
        return xr.apply_ufunc(
            np.linalg.norm,
            data,
            input_core_dims=[["space"]],
            kwargs={"axis": -1},
        )
    elif "space_pol" in data.dims:
        # Allow both 2D polar and 3D cylindrical
        if len(data.coords["space_pol"]) == 2:
            validate_dims_coords(data, {"space_pol": ["rho", "phi"]})
        elif len(data.coords["space_pol"]) == 3:
            validate_dims_coords(data, {"space_pol": ["rho", "phi", "z"]})
        else:
            _raise_error_for_invalid_spatial_dim_length("space_pol", 2, 3)
        return data.sel(space_pol="rho", drop=True)
    else:
        _raise_error_for_missing_spatial_dim()


def convert_to_unit(data: xr.DataArray) -> xr.DataArray:
    """Convert the vectors along the spatial dimension into unit vectors.

    A unit vector is a vector pointing in the same direction as the original
    vector but with norm = 1.

    Parameters
    ----------
    data
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
        # Allow both 2D and 3D
        if len(data.coords["space"]) == 2:
            validate_dims_coords(data, {"space": ["x", "y"]})
        elif len(data.coords["space"]) == 3:
            validate_dims_coords(data, {"space": ["x", "y", "z"]})
        else:
            _raise_error_for_invalid_spatial_dim_length("space", 2, 3)
        return data / compute_norm(data)
    elif "space_pol" in data.dims:
        # Allow both 2D polar and 3D cylindrical
        if len(data.coords["space_pol"]) == 2:
            validate_dims_coords(data, {"space_pol": ["rho", "phi"]})
        elif len(data.coords["space_pol"]) == 3:
            validate_dims_coords(data, {"space_pol": ["rho", "phi", "z"]})
        else:
            _raise_error_for_invalid_spatial_dim_length("space_pol", 2, 3)
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
    """Transform Cartesian coordinates to polar (2D) or cylindrical (3D).

    Parameters
    ----------
    data
        The input data containing ``space`` as a dimension,
        with ``x`` and ``y`` (2D) or ``x``, ``y``, and ``z`` (3D)
        in the dimension coordinate.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the polar/cylindrical coordinates
        stored in the ``space_pol`` dimension:

        - 2D: ``rho`` and ``phi``
        - 3D: ``rho``, ``phi``, and ``z`` (cylindrical coordinates)

        The angle ``phi`` is in radians, in the range ``[-pi, pi]``.
        For 3D input, ``z`` is passed through unchanged.

    Notes
    -----
    To compute the angle ``phi`` we rely on the :obj:`numpy.arctan2`
    function, which follows the C standard [1]_. The C standard considers
    the case in which the inputs to the ``arctan2`` [2]_ function are signed
    zeros [3]_. For simplicity and interpretability, in ``movement`` we
    only consider the case of unsigned (positive) zeros. We implement it
    by setting the angle ``phi`` to 0 when the norm ``rho`` of the vector is 0.

    References
    ----------
    .. [1] ISO/IEC standard 9899:1999, "Programming language C."
    .. [2] https://en.wikipedia.org/wiki/Atan2
    .. [3] https://en.wikipedia.org/wiki/Signed_zero

    See Also
    --------
    :obj:`numpy.arctan2`

    """
    # Validate 2D or 3D input
    is_3d = len(data.coords["space"]) == 3
    if is_3d:
        validate_dims_coords(data, {"space": ["x", "y", "z"]})
    else:
        validate_dims_coords(data, {"space": ["x", "y"]})

    x = data.sel(space="x")
    y = data.sel(space="y")
    rho = np.sqrt(x**2 + y**2)
    phi = xr.apply_ufunc(np.arctan2, y, x)

    # Make all zeros in phi positive zeros
    # - where rho == 0, set phi to 0
    # - where rho != 0, keep the phi value from atan2
    phi = xr.where(np.isclose(rho.values, 0.0, atol=1e-9), 0.0, phi)

    # Build output components
    components = [
        rho.assign_coords({"space_pol": "rho"}),
        phi.assign_coords({"space_pol": "phi"}),
    ]

    # For 3D, pass z through unchanged
    if is_3d:
        z = data.sel(space="z")
        components.append(z.assign_coords({"space_pol": "z"}))

    # Replace space dim with space_pol
    dims = list(data.dims)
    dims[dims.index("space")] = "space_pol"
    return xr.concat(components, dim="space_pol").transpose(*dims)


def pol2cart(data: xr.DataArray) -> xr.DataArray:
    """Transform polar (2D) or cylindrical (3D) coordinates to Cartesian.

    Parameters
    ----------
    data
        The input data containing ``space_pol`` as a dimension,
        with ``rho`` and ``phi`` (2D) or ``rho``, ``phi``, and ``z`` (3D)
        in the dimension coordinate.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the Cartesian coordinates
        stored in the ``space`` dimension:

        - 2D: ``x`` and ``y``
        - 3D: ``x``, ``y``, and ``z``

    """
    # Validate 2D or 3D input
    is_3d = len(data.coords["space_pol"]) == 3
    if is_3d:
        validate_dims_coords(data, {"space_pol": ["rho", "phi", "z"]})
    else:
        validate_dims_coords(data, {"space_pol": ["rho", "phi"]})

    rho = data.sel(space_pol="rho")
    phi = data.sel(space_pol="phi")
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    # Build output components
    components = [
        x.assign_coords({"space": "x"}),
        y.assign_coords({"space": "y"}),
    ]

    # For 3D, pass z through unchanged
    if is_3d:
        z = data.sel(space_pol="z")
        components.append(z.assign_coords({"space": "z"}))

    # Replace space_pol dim with space
    dims = list(data.dims)
    dims[dims.index("space_pol")] = "space"
    return xr.concat(components, dim="space").transpose(*dims)


def cart2sph(data: xr.DataArray) -> xr.DataArray:
    """Transform 3D Cartesian coordinates to spherical.

    Parameters
    ----------
    data
        The input data containing ``space`` as a dimension,
        with ``x``, ``y``, and ``z`` in the dimension coordinate.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the spherical coordinates
        stored in the ``space_sph`` dimension, with ``rho``,
        ``azimuth``, and ``elevation`` in the dimension coordinate:

        - ``rho``: radial distance (magnitude of the vector)
        - ``azimuth``: angle in the x-y plane from the positive x-axis,
          in radians, in the range ``[-pi, pi]``
        - ``elevation``: angle from the x-y plane, in radians,
          in the range ``[-pi/2, pi/2]``

    See Also
    --------
    sph2cart : Inverse transformation from spherical to Cartesian.

    """
    validate_dims_coords(data, {"space": ["x", "y", "z"]})

    x = data.sel(space="x")
    y = data.sel(space="y")
    z = data.sel(space="z")

    rho = np.sqrt(x**2 + y**2 + z**2)
    azimuth = xr.apply_ufunc(np.arctan2, y, x)
    # Compute elevation, handling zero-magnitude vectors
    elevation = xr.where(
        rho > 0,
        np.arcsin((z / rho).clip(-1, 1)),
        0.0,
    )

    # Replace space dim with space_sph
    dims = list(data.dims)
    dims[dims.index("space")] = "space_sph"
    return xr.concat(
        [
            rho.assign_coords({"space_sph": "rho"}),
            azimuth.assign_coords({"space_sph": "azimuth"}),
            elevation.assign_coords({"space_sph": "elevation"}),
        ],
        dim="space_sph",
    ).transpose(*dims)


def sph2cart(data: xr.DataArray) -> xr.DataArray:
    """Transform spherical coordinates to 3D Cartesian.

    Parameters
    ----------
    data
        The input data containing ``space_sph`` as a dimension,
        with ``rho``, ``azimuth``, and ``elevation`` in the
        dimension coordinate.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the Cartesian coordinates
        stored in the ``space`` dimension, with ``x``, ``y``, and ``z``
        in the dimension coordinate.

    See Also
    --------
    cart2sph : Inverse transformation from Cartesian to spherical.

    """
    validate_dims_coords(data, {"space_sph": ["rho", "azimuth", "elevation"]})

    rho = data.sel(space_sph="rho")
    azimuth = data.sel(space_sph="azimuth")
    elevation = data.sel(space_sph="elevation")

    x = rho * np.cos(elevation) * np.cos(azimuth)
    y = rho * np.cos(elevation) * np.sin(azimuth)
    z = rho * np.sin(elevation)

    # Replace space_sph dim with space
    dims = list(data.dims)
    dims[dims.index("space_sph")] = "space"
    return xr.concat(
        [
            x.assign_coords({"space": "x"}),
            y.assign_coords({"space": "y"}),
            z.assign_coords({"space": "z"}),
        ],
        dim="space",
    ).transpose(*dims)


def compute_signed_angle_2d(
    u: xr.DataArray,
    v: xr.DataArray | np.ndarray,
    v_as_left_operand: bool = False,
) -> xr.DataArray:
    r"""Compute the signed angle from the vector ``u`` to the vector ``v``.

    The signed angle between ``u`` and ``v`` is the rotation that needs to be
    applied to ``u`` to have it point in the same direction as ``v`` (see
    Notes). Angles are returned in radians, spanning :math:`(-\pi, \pi]`
    according to the ``arctan2`` convention.

    Parameters
    ----------
    u
        An array of position vectors containing the ``space``
        dimension with only ``"x"`` and ``"y"`` coordinates.
    v
        A 2D vector (or array of 2D vectors) against which to
        compare ``u``. May either be an xarray
        DataArray containing the ``"space"``  dimension or a numpy
        array containing one or more 2D vectors. (See Notes)
    v_as_left_operand
        If True, the signed angle between ``v`` and ``u`` is returned, instead
        of the signed angle between ``u`` and ``v``. This is a convenience
        wrapper for when one of the two vectors to be used does not have time
        points, and the other does.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing signed angle between
        ``u`` and ``v`` for every time point. Matches the dimensions of
        ``u``, but without the ``space`` dimension.

    Notes
    -----
    Given two vectors :math:`u = (u_x, u_y)` and :math:`v = (v_x, v_y)`,
    the signed angle :math:`\alpha` between ``u`` and ``v`` is computed as

    .. math::
        \alpha &= \mathrm{arctan2}(u \times v, u\cdot v) \\
        &= \mathrm{arctan2}(u_x v_y - u_y v_x, u_x v_x + u_y v_y),

    which corresponds to the rotation that needs to be applied to ``u`` for it
    to point in the direction of ``v``.

    If ``v`` is passed as an ``xarray.DataArray``, ``v`` must have spatial
    coordinates that match those of ``u``. Furthermore, any dimensions that are
    present in both ``u`` and ``v`` must match in length.

    If passed as a numpy array, ``v`` must have one of three shapes:

    - ``(2,)``: where dimension ``0`` contains spatial
      coordinates (x,y), and no time dimension is specified.
    - ``(1,2)``:, where dimension ``0`` corresponds to a
      single time-point and dimension ``1`` contains spatial
      coordinates (x,y).
    - ``(n,2)``: where dimension ``0`` corresponds to
      time and dimension ``1`` contains spatial coordinates
      (x,y), and where ``n == len(u.time)``.

    Vectors given as ``v`` that contain more dimensions, or have shapes
    otherwise different from those defined above are considered invalid.

    """
    validate_dims_coords(u, {"space": ["x", "y"]}, exact_coords=True)
    # Ensure v can be broadcast over u
    if isinstance(v, np.ndarray):
        v = v.squeeze()
        if v.ndim == 1:
            v_dims = ["space"]
        elif v.ndim == 2:
            v_dims = ["time", "space"]
        else:
            raise logger.error(
                ValueError(f"v must be 1D or 2D, but got {v.ndim}D.")
            )
        v = xr.DataArray(
            v,
            dims=v_dims,
            coords={d: u.coords[d] for d in v_dims},
        )
    elif not isinstance(v, xr.DataArray):
        raise logger.error(
            TypeError(
                "v must be an xarray.DataArray or np.ndarray, "
                f"but got {type(v)}."
            )
        )
    validate_dims_coords(v, {"space": ["x", "y"]}, exact_coords=True)

    u = convert_to_unit(u)
    u_x = u.sel(space="x")
    u_y = u.sel(space="y")

    v = convert_to_unit(v)
    v_x = v.sel(space="x")
    v_y = v.sel(space="y")

    cross = u_x * v_y - u_y * v_x
    if v_as_left_operand:
        cross *= -1.0
    dot = u_x * v_x + u_y * v_y

    angles = np.arctan2(cross, dot)
    assert isinstance(angles, xr.DataArray)
    # arctan2 returns values in [-pi, pi].
    # We need to map -pi angles to pi, to stay in the (-pi, pi] range
    angles.values[angles <= -np.pi] = np.pi
    angles.name = "signed_angle"
    return angles


def _raise_error_for_missing_spatial_dim() -> NoReturn:
    raise logger.error(
        ValueError(
            "Input data array must contain either 'space' or 'space_pol' "
            "as dimensions."
        )
    )


def _raise_error_for_invalid_spatial_dim_length(
    dim_name: str, *valid_lengths: int
) -> NoReturn:
    lengths_str = " or ".join(str(n) for n in valid_lengths)
    raise logger.error(
        ValueError(
            f"Dimension '{dim_name}' must have {lengths_str} coordinates."
        )
    )
