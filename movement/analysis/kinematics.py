import numpy as np
import xarray as xr


def displacement(data: xr.DataArray) -> xr.Dataset:
    """Compute the displacement between consecutive locations
    of each keypoint of each individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where
        the last dimension contains the x and y coordinates.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing the computed magnitude and
        direction of displacement.
    """
    displacement_xy = data.diff(dim="time")
    displacement_xy = displacement_xy.reindex(data.coords, fill_value=0)
    return compute_vector_magnitude_direction(displacement_xy)


def distance(data: xr.DataArray) -> xr.DataArray:
    """Compute the Euclidean distances between consecutive
    locations of each keypoint of each individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where
        the last dimension contains the x and y coordinates.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the magnitude of displacement.
    """
    return displacement(data).magnitude


def velocity(data: xr.DataArray) -> xr.Dataset:
    """Compute the velocity of a single keypoint from
    a single individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing the computed magnitude and
        direction of velocity.
    """
    return approximate_derivative(data, order=1)


def speed(data: xr.DataArray) -> xr.DataArray:
    """Compute speed based on the Euclidean norm (magnitude) of the
    differences between consecutive points, i.e. the straight-line
    distance travelled, assuming equidistant time spacing.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the magnitude of velocity.
    """
    return velocity(data).magnitude


def acceleration(data: xr.DataArray) -> xr.Dataset:
    """Compute the acceleration of a single keypoint from
    a single individual.

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
    return approximate_derivative(data, order=2)


def approximate_derivative(data: xr.DataArray, order: int = 1) -> xr.Dataset:
    """Compute velocity or acceleration using numerical differentiation,
    assuming equidistant time spacing.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.
    order : int
        The order of the derivative. 1 for velocity, 2 for
        acceleration. Default is 1.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing the computed magnitudes and
        directions of the derived variable.
    """
    if order <= 0:
        raise ValueError("order must be a positive integer.")
    else:
        result = data
        dt = data["time"].diff(dim="time").values[0]
        for _ in range(order):
            result = xr.apply_ufunc(
                np.gradient,
                result,
                dt,
                kwargs={"axis": 0},
            )
        result = result.reindex_like(data)
    return compute_vector_magnitude_direction(result)


def compute_vector_magnitude_direction(input: xr.DataArray) -> xr.Dataset:
    """Compute the magnitude and direction of a vector.

    Parameters
    ----------
    input : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing the computed magnitude and
        direction.
    """
    magnitude = xr.apply_ufunc(
        np.linalg.norm,
        input,
        input_core_dims=[["space"]],
        kwargs={"axis": -1},
    )
    direction = xr.apply_ufunc(
        np.arctan2,
        input[..., 1],
        input[..., 0],
    )
    return xr.Dataset({"magnitude": magnitude, "direction": direction})


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
