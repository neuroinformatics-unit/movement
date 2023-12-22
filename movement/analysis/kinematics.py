import numpy as np
import xarray as xr


def compute_velocity(
    data: xr.DataArray, method: str = "euclidean"
) -> np.ndarray:
    """Compute the instantaneous velocity of a single keypoint from
    a single individual.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.
    method : str
        The method to use for computing velocity. Can be "euclidean" or
        "numerical".

    Returns
    -------
    numpy.ndarray
        A numpy array containing the computed velocity.
    """
    if method == "euclidean":
        return compute_euclidean_velocity(data)
    return approximate_derivative(data, order=1)


def compute_euclidean_velocity(data: xr.DataArray) -> np.ndarray:
    """Compute velocity based on the Euclidean norm (magnitude) of the
    differences between consecutive points, i.e. the straight-line
    distance travelled.

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
    time_diff = data["time"].diff(dim="time")
    space_diff = np.linalg.norm(np.diff(data.values, axis=0), axis=1)
    velocity = space_diff / time_diff
    # Pad with zero to match the original shape of the data
    velocity = np.concatenate([np.zeros((1,) + velocity.shape[1:]), velocity])
    return velocity


def approximate_derivative(data: xr.DataArray, order: int = 0) -> np.ndarray:
    """Compute displacement, velocity, or acceleration using numerical
    differentiation, assuming equidistant time spacing.

    Parameters
    ----------
    data : xarray.DataArray
        The input data, assumed to be of shape (..., 2), where the last
        dimension contains the x and y coordinates.
    order : int
        The order of the derivative. 0 for displacement, 1 for velocity, 2 for
        acceleration.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the computed kinematic variable.
    """
    if order == 0:  # Compute displacement
        result = np.diff(data, axis=0)
        # Pad with zeros to match the original shape of the data
        result = np.concatenate([np.zeros((1,) + result.shape[1:]), result])
    else:
        result = data
        dt = data["time"].diff(dim="time").values[0]
        for _ in range(order):
            result = np.gradient(result, dt, axis=0)
    magnitude = np.linalg.norm(result, axis=-1)
    # Pad with zero to match the output of compute_euclidean_velocity
    magnitude = np.pad(magnitude[:-1], (1, 0), "constant")
    # direction = np.arctan2(result[..., 1], result[..., 0])
    return magnitude


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
