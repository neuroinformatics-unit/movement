"""Utility functions for vector operations."""

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
    u : xarray.DataArray
        An array of position vectors containing the ``space``
        dimension with only ``"x"`` and ``"y"`` coordinates.
    v : xarray.DataArray | numpy.ndarray
        A 2D vector (or array of 2D vectors) against which to
        compare ``u``. May either be an xarray
        DataArray containing the ``"space"``  dimension or a numpy
        array containing one or more 2D vectors. (See Notes)
    v_as_left_operand : bool, default False
        If True, the signed angle between ``v`` and ``u`` is returned, instead
        of the signed angle between ``u`` and ``v``. This is a convenience
        wrapper for when one of the two vectors to be used does not have time
        points, and the other does.

    Returns
    -------
    xarray.DataArray :
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
    # arctan2 returns values in [-pi, pi].
    # We need to map -pi angles to pi, to stay in the (-pi, pi] range
    angles.values[angles <= -np.pi] = np.pi
    angles.name = "signed_angle"
    return angles


def _raise_error_for_missing_spatial_dim() -> None:
    raise logger.error(
        ValueError(
            "Input data array must contain either 'space' or 'space_pol' "
            "as dimensions."
        )
    )
