import numpy as np
import xarray as xr


def compute_displacement(data: xr.DataArray) -> xr.DataArray:
    """Compute the displacement between consecutive x, y
    locations of each keypoint of each individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where
        the last dimension contains the x and y coordinates.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed displacement.
    """
    displacement_xy = data.diff(dim="time")
    displacement_xy = displacement_xy.reindex(data.coords, fill_value=0)
    return displacement_xy


def compute_velocity(data: xr.DataArray) -> xr.DataArray:
    """Compute the velocity between consecutive x, y locations
    of each keypoint of each individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.

    Returns
    -------
    xarray.DataArray
        An xarray Dataset containing the computed velocity.
    """
    return compute_approximate_derivative(data, order=1)


def compute_acceleration(data: xr.DataArray) -> xr.DataArray:
    """Compute the acceleration between consecutive x, y
    locations of each keypoint of each individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing the magnitude and direction
        of acceleration.
    """
    return compute_approximate_derivative(data, order=2)


def compute_approximate_derivative(
    data: xr.DataArray, order: int = 1
) -> xr.DataArray:
    """Compute velocity or acceleration using numerical differentiation,
    assuming equidistant time spacing.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains data in the x and y dimensions.
    order : int
        The order of the derivative. 1 for velocity, 2 for
        acceleration. Default is 1.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the derived variable.
    """
    if order <= 0:
        raise ValueError("order must be a positive integer.")
    else:
        result = data
        dt = data["time"].values[1] - data["time"].values[0]
        for _ in range(order):
            result = xr.apply_ufunc(
                np.gradient,
                result,
                dt,
                kwargs={"axis": 0},
            )
        result = result.reindex_like(data)
    return result


def compute_norm(data: xr.DataArray) -> xr.DataArray:
    """Compute the Euclidean norm (magnitude) of a vector.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains data in the x and y dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed Euclidean norm.
    """
    return xr.apply_ufunc(
        np.linalg.norm,
        data,
        input_core_dims=[["space"]],
        kwargs={"axis": -1},
    )


def compute_theta(data: xr.DataArray) -> xr.DataArray:
    """Compute the theta (direction) of a vector.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains data in the x and y dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed theta.
    """
    return xr.apply_ufunc(
        np.arctan2,
        data[..., 1],
        data[..., 0],
    )


# Locomotion Features
# speed
# speed_centroid
# acceleration
# acceleration_centroid
# speed_fwd
# radial_vel
# tangential_vel
# speed_centroid_w(s)
# speed_(p)_w(s)
