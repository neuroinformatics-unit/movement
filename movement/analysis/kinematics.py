import numpy as np
import xarray as xr


def displacement(data: xr.DataArray) -> np.ndarray:
    """Compute the displacement between consecutive locations
    of a single keypoint from a single individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the computed magnitude and
        direction of the displacement.
    """
    displacement_vector = np.diff(data, axis=0, prepend=data[0:1])
    magnitude = np.linalg.norm(displacement_vector, axis=1)
    direction = np.arctan2(
        displacement_vector[..., 1], displacement_vector[..., 0]
    )
    return np.stack((magnitude, direction), axis=1)


def distance(data: xr.DataArray) -> np.ndarray:
    """Compute the distances between consecutive locations of
    a single keypoint from a single individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the computed distance.
    """
    return displacement(data)[:, 0]


def velocity(data: xr.DataArray) -> np.ndarray:
    """Compute the velocity of a single keypoint from
    a single individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the computed velocity.
    """
    return approximate_derivative(data, order=1)


def speed(data: xr.DataArray) -> np.ndarray:
    """Compute velocity based on the Euclidean norm (magnitude) of the
    differences between consecutive points, i.e. the straight-line
    distance travelled, assuming equidistant time spacing.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the computed velocity.
    """
    return velocity(data)[:, 0]


def acceleration(data: xr.DataArray) -> np.ndarray:
    """Compute the acceleration of a single keypoint from
    a single individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the computed acceleration.
    """
    return approximate_derivative(data, order=2)


def approximate_derivative(data: xr.DataArray, order: int = 1) -> np.ndarray:
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
    numpy.ndarray
        A numpy array containing the computed magnitudes and directions of
        the kinematic variable.
    """
    if order <= 0:
        raise ValueError("order must be a positive integer.")
    else:
        result = data
        dt = data["time"].diff(dim="time").values[0]
        for _ in range(order):
            result = np.gradient(result, dt, axis=0)
        # Prepend with zeros to match match output to the input shape
        result = np.pad(result[1:], ((1, 0), (0, 0)), "constant")
    magnitude = np.linalg.norm(result, axis=-1)
    direction = np.arctan2(result[..., 1], result[..., 0])
    return np.stack((magnitude, direction), axis=1)


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
