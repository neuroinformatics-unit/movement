"""Compute trajectory complexity measures.

This module provides functions to compute various measures of trajectory
complexity, which quantify how straight or tortuous a path is. These metrics
are useful for analyzing animal movement patterns across space.
"""

from typing import Literal

import numpy as np
import xarray as xr

from movement.kinematics import compute_displacement, compute_path_length
from movement.utils.logging import log_error, log_to_attrs, log_warning
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


@log_to_attrs
def compute_straightness_index(
    data: xr.DataArray,
    start: float | None = None,
    stop: float | None = None,
) -> xr.DataArray:
    """Compute the straightness index of a trajectory.

    The straightness index is defined as the ratio of the Euclidean distance
    between the start and end points of a trajectory to the total path length.
    Values range from 0 to 1, where 1 indicates a perfectly straight path,
    and values closer to 0 indicate more tortuous paths.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    start : float, optional
        The start time of the trajectory. If None (default),
        the minimum time coordinate in the data is used.
    stop : float, optional
        The end time of the trajectory. If None (default),
        the maximum time coordinate in the data is used.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed straightness index,
        with dimensions matching those of the input data,
        except ``time`` and ``space`` are removed.

    Notes
    -----
    The straightness index (SI) is calculated as:

    SI = Euclidean distance / Path length

    where the Euclidean distance is the straight-line distance between the
    start and end points, and the path length is the total distance traveled
    along the trajectory.

    References
    ----------
    .. [1] Batschelet, E. (1981). Circular statistics in biology.
           London: Academic Press.

    """
    validate_dims_coords(data, {"time": [], "space": []})

    # Determine start and stop points
    if start is None:
        start = data.time.min().item()
    if stop is None:
        stop = data.time.max().item()

    # Extract start and end positions
    start_pos = data.sel(time=start, method="nearest")
    end_pos = data.sel(time=stop, method="nearest")

    # Calculate Euclidean distance between start and end points
    euclidean_distance = compute_norm(end_pos - start_pos)

    # Calculate path length
    path_length = compute_path_length(data, start=start, stop=stop)

    # Compute straightness index
    straightness_index = euclidean_distance / path_length

    return straightness_index


@log_to_attrs
def compute_sinuosity(
    data: xr.DataArray,
    window_size: int = 10,
    stride: int = 1,
) -> xr.DataArray:
    """Compute the sinuosity of a trajectory using a sliding window.

    Sinuosity is computed as the ratio of the path length to the Euclidean distance
    between the start and end points, within each window. This provides a
    local measure of trajectory complexity that varies along the path.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    window_size : int, optional
        The size of the sliding window in number of time points.
        Default is 10.
    stride : int, optional
        The step size for the sliding window. Default is 1.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed sinuosity at each time point,
        with dimensions matching those of the input data,
        except ``space`` is removed.

    Notes
    -----
    The sinuosity is essentially the inverse of the straightness index.
    Values range from 1 to infinity, where 1 indicates a perfectly straight path,
    and higher values indicate more tortuous paths.

    References
    ----------
    .. [1] Benhamou, S. (2004). How to reliably estimate the tortuosity of
           an animal's path: straightness, sinuosity, or fractal dimension?
           Journal of Theoretical Biology, 229(2), 209-220.

    """
    validate_dims_coords(data, {"time": [], "space": []})

    # Validate window_size
    if window_size < 2:
        raise log_error(
            ValueError,
            "window_size must be at least 2 to compute sinuosity.",
        )

    # Get number of time points
    n_time = data.sizes["time"]

    # Initialize result array with NaNs
    result = xr.full_like(data.isel(space=0), fill_value=np.nan)

    # Calculate sinuosity for each window
    for i in range(0, n_time - window_size + 1, stride):
        # Extract window data
        window_data = data.isel(time=slice(i, i + window_size))

        # Extract start and end positions
        start_pos = window_data.isel(time=0)
        end_pos = window_data.isel(time=-1)

        # Calculate Euclidean distance between start and end points
        euclidean_distance = compute_norm(end_pos - start_pos)

        # Calculate path length within window
        displacements = compute_displacement(window_data).isel(
            time=slice(1, None)
        )
        path_length = compute_norm(displacements).sum(dim="time")

        # Compute sinuosity (inverse of straightness)
        sinuosity = path_length / euclidean_distance

        # Assign to middle point of window
        mid_idx = i + window_size // 2
        result.isel(time=mid_idx).data = sinuosity.data

    return result


@log_to_attrs
def compute_angular_velocity(
    data: xr.DataArray,
    in_degrees: bool = False,
) -> xr.DataArray:
    """Compute the angular velocity of a trajectory.

    Angular velocity measures the rate of change of the angle of movement.
    It is computed as the angle between consecutive displacement vectors.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    in_degrees : bool, optional
        Whether to return the result in degrees (True) or radians (False).
        Default is False.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed angular velocity,
        with dimensions matching those of the input data,
        except ``space`` is removed.

    Notes
    -----
    Angular velocity is defined as the angle between consecutive displacement
    vectors divided by the time interval. High angular velocities indicate
    sharp turns or changes in direction.

    """
    validate_dims_coords(data, {"time": [], "space": []})

    # Compute displacement vectors
    displacement = compute_displacement(data)

    # Skip first time point (displacement is 0)
    displacement = displacement.isel(time=slice(1, None))

    # Compute unit vectors (normalize displacement)
    unit_displacement = displacement / compute_norm(displacement).fillna(1)

    # Compute dot products between consecutive unit vectors
    dot_products = (
        unit_displacement.isel(time=slice(1, None))
        * unit_displacement.isel(time=slice(0, -1))
    ).sum(dim="space")

    # Clip dot products to [-1, 1] to handle numerical errors
    dot_products = xr.where(dot_products > 1, 1, dot_products)
    dot_products = xr.where(dot_products < -1, -1, dot_products)

    # Compute angles in radians
    angles = np.arccos(dot_products)

    # Convert to degrees if requested
    if in_degrees:
        angles = np.rad2deg(angles)

    # Create result array with same dimensions as input but with NaNs at endpoints
    result = xr.full_like(data.isel(space=0), fill_value=np.nan)

    # Assign computed angles to result (offset by 1 to account for displacement calculation)
    result.isel(time=slice(2, None)).data = angles.data

    return result


@log_to_attrs
def compute_tortuosity(
    data: xr.DataArray,
    start: float | None = None,
    stop: float | None = None,
    method: Literal["fractal", "angular_variance"] = "angular_variance",
    window_size: int = 10,
) -> xr.DataArray:
    """Compute the tortuosity of a trajectory.

    Tortuosity is a measure of the degree of winding or twisting of a path.
    This function provides multiple methods to compute tortuosity.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    start : float, optional
        The start time of the trajectory. If None (default),
        the minimum time coordinate in the data is used.
    stop : float, optional
        The end time of the trajectory. If None (default),
        the maximum time coordinate in the data is used.
    method : Literal["fractal", "angular_variance"], optional
        The method to use for computing tortuosity.
        "fractal" uses box-counting fractal dimension.
        "angular_variance" uses the circular variance of turning angles.
        Default is "angular_variance".
    window_size : int, optional
        The size of the window used for the fractal method.
        Default is 10. Only used if method="fractal".

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed tortuosity,
        with dimensions matching those of the input data,
        except ``time`` and ``space`` are removed.

    Notes
    -----
    The "fractal" method estimates the fractal dimension of the path using
    the box-counting method. It ranges from 1 (straight line) to 2 (completely
    space-filling curve).

    The "angular_variance" method computes the circular variance of turning
    angles. It ranges from 0 (straight line) to 1 (highly tortuous path with
    uniformly distributed turning angles).

    References
    ----------
    .. [1] Nams, V. O. (1996). The VFractal: a new estimator for fractal
           dimension of animal movement paths. Landscape Ecology, 11(5), 289-297.
    .. [2] Benhamou, S. (2004). How to reliably estimate the tortuosity of
           an animal's path: straightness, sinuosity, or fractal dimension?
           Journal of Theoretical Biology, 229(2), 209-220.

    """
    validate_dims_coords(data, {"time": [], "space": []})

    # Determine start and stop points
    if start is None:
        start = data.time.min().item()
    if stop is None:
        stop = data.time.max().item()

    # Filter data to desired time range
    data_filtered = data.sel(time=slice(start, stop))

    if method == "angular_variance":
        # Compute displacement vectors
        displacement = compute_displacement(data_filtered)

        # Skip first time point (displacement is 0)
        displacement = displacement.isel(time=slice(1, None))

        # Compute unit vectors (normalize displacement)
        unit_displacement = displacement / compute_norm(displacement).fillna(1)

        # Compute dot products between consecutive unit vectors
        dot_products = (
            unit_displacement.isel(time=slice(1, None))
            * unit_displacement.isel(time=slice(0, -1))
        ).sum(dim="space")

        # Clip dot products to [-1, 1] to handle numerical errors
        dot_products = xr.where(dot_products > 1, 1, dot_products)
        dot_products = xr.where(dot_products < -1, -1, dot_products)

        # Compute angles in radians
        angles = np.arccos(dot_products)

        # Compute circular mean of cosines and sines of angles
        mean_cos = np.cos(angles).mean(dim="time")
        mean_sin = np.sin(angles).mean(dim="time")

        # Compute circular variance (R = 1 - mean resultant length)
        R = 1 - np.sqrt(mean_cos**2 + mean_sin**2)

        return R

    elif method == "fractal":
        # Implementing a simplified box-counting fractal dimension
        if len(data_filtered.space) != 2:
            raise log_error(
                ValueError,
                "The fractal dimension method only works with 2D data.",
            )

        # Get x and y coordinates
        x = data_filtered.sel(space="x").values
        y = data_filtered.sel(space="y").values

        # Normalize to [0, 1] range
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

        # Initialize arrays for box counts
        scales = []
        counts = []

        # Calculate box counts at different scales
        for scale in range(2, min(64, len(x_norm) // 4)):
            # Create grid
            grid_size = 1.0 / scale
            occupied_boxes = set()

            # Count occupied boxes
            for i in range(len(x_norm)):
                box_x = int(x_norm[i] / grid_size)
                box_y = int(y_norm[i] / grid_size)
                occupied_boxes.add((box_x, box_y))

            scales.append(scale)
            counts.append(len(occupied_boxes))

        if len(scales) < 2:
            log_warning(
                "Not enough data points to compute fractal dimension. Returning NaN."
            )
            result = xr.full_like(
                data_filtered.isel(time=0, space=0, drop=True),
                fill_value=np.nan,
            )
            return result

        # Compute fractal dimension as the slope of log(count) vs log(scale)
        log_scales = np.log(scales)
        log_counts = np.log(counts)

        # Linear regression: log(count) = D * log(scale) + b
        D = np.polyfit(log_scales, log_counts, 1)[0]

        # Create result array
        # Remove time and space dimensions
        dims = [
            dim for dim in data_filtered.dims if dim not in ["time", "space"]
        ]
        coords = {dim: data_filtered[dim] for dim in dims}

        result = xr.DataArray(
            D,
            dims=dims,
            coords=coords,
        )

        return result
    else:
        raise log_error(
            ValueError,
            f"Unknown method: {method}. Use 'fractal' or 'angular_variance'.",
        )


@log_to_attrs
def compute_directional_change(
    data: xr.DataArray,
    window_size: int = 10,
    in_degrees: bool = False,
) -> xr.DataArray:
    """Compute the directional change along a trajectory.

    Directional change measures the total amount of turning within a window,
    calculated as the sum of absolute angular changes between consecutive
    displacement vectors.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    window_size : int, optional
        The size of the sliding window in number of time points.
        Default is 10.
    in_degrees : bool, optional
        Whether to return the result in degrees (True) or radians (False).
        Default is False.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed directional change at each time point,
        with dimensions matching those of the input data,
        except ``space`` is removed.

    Notes
    -----
    Directional change is calculated as the sum of absolute angular changes
    within a sliding window. Higher values indicate more turning or meandering
    behavior, while lower values indicate more directed movement.

    """
    validate_dims_coords(data, {"time": [], "space": []})

    # Compute angular velocity (in radians)
    angular_velocity = compute_angular_velocity(data, in_degrees=False)

    # Get number of time points
    n_time = data.sizes["time"]

    # Initialize result array with NaNs
    result = xr.full_like(data.isel(space=0), fill_value=np.nan)

    # Calculate directional change for each window
    for i in range(0, n_time - window_size + 1):
        # Extract window data
        window_data = angular_velocity.isel(time=slice(i, i + window_size))

        # Sum absolute angular changes
        directional_change = np.nansum(np.abs(window_data.values))

        # Convert to degrees if requested
        if in_degrees:
            directional_change = np.rad2deg(directional_change)

        # Assign to middle point of window
        mid_idx = i + window_size // 2
        result.isel(time=mid_idx).data = directional_change

    return result
