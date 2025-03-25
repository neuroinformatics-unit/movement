"""Combine tracking data from multiple sources.

This module provides functions for combining tracking data from multiple
sources to produce more accurate trajectories. This is particularly useful
in cases where different tracking methods may fail in different situations,
such as in multi-animal tracking with ID swaps.
"""

import logging
from enum import Enum, auto
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy.signal import medfilt

from movement.filtering import interpolate_over_time, rolling_filter
from movement.utils.logging import log_error, log_to_attrs
from movement.utils.reports import report_nan_values
from movement.validators.arrays import validate_dims_coords

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Enumeration of available track fusion methods."""

    MEAN = auto()
    MEDIAN = auto()
    WEIGHTED = auto()
    RELIABILITY_BASED = auto()
    KALMAN = auto()


@log_to_attrs
def align_datasets(
    datasets: List[xr.Dataset],
    keypoint: str = "centroid",
    interpolate: bool = True,
    max_gap: Optional[int] = 5,
) -> List[xr.DataArray]:
    """Aligns multiple datasets to have the same time coordinates.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of datasets containing position data to align.
    keypoint : str, optional
        The keypoint to extract from each dataset, by default "centroid".
    interpolate : bool, optional
        Whether to interpolate missing values after alignment, by default True.
    max_gap : int, optional
        Maximum size of gap to interpolate, by default 5.

    Returns
    -------
    list of xarray.DataArray
        List of aligned DataArrays containing only the specified keypoint position data.

    Notes
    -----
    This function extracts the specified keypoint from each dataset, aligns them to
    have the same time coordinates, and optionally interpolates missing values.
    """
    if not datasets:
        raise log_error(ValueError, "No datasets provided")

    # Extract the keypoint position data from each dataset
    position_arrays = []
    for ds in datasets:
        # Check if keypoint exists in this dataset
        if "keypoints" in ds.dims and keypoint not in ds.keypoints.values:
            available_keypoints = list(ds.keypoints.values)
            raise log_error(
                ValueError,
                f"Keypoint '{keypoint}' not found in dataset. "
                f"Available keypoints: {available_keypoints}",
            )
        
        # Extract position for this keypoint
        if "keypoints" in ds.dims:
            pos = ds.position.sel(keypoints=keypoint)
        else:
            # Handle datasets without keypoints dimension
            pos = ds.position
        
        position_arrays.append(pos)

    # Get union of all time coordinates
    all_times = sorted(set().union(*[set(arr.time.values) for arr in position_arrays]))
    
    # Reindex all arrays to the common time coordinate
    aligned_arrays = []
    for arr in position_arrays:
        reindexed = arr.reindex(time=all_times)
        
        # Optionally interpolate missing values
        if interpolate:
            reindexed = interpolate_over_time(reindexed, max_gap=max_gap)
            
        aligned_arrays.append(reindexed)
    
    return aligned_arrays


@log_to_attrs
def fuse_tracks_mean(
    aligned_tracks: List[xr.DataArray],
    print_report: bool = False,
) -> xr.DataArray:
    """Fuse tracks by taking the mean across all sources.

    Parameters
    ----------
    aligned_tracks : list of xarray.DataArray
        List of aligned position DataArrays.
    print_report : bool, optional
        Whether to print a report on the number of NaNs in the result, by default False.

    Returns
    -------
    xarray.DataArray
        Fused track with position values averaged across sources.

    Notes
    -----
    This function computes the mean of all valid position values at each time point.
    If all sources have NaN at a particular time point, the result will also be NaN.
    """
    if not aligned_tracks:
        raise log_error(ValueError, "No tracks provided")

    # Stack all tracks along a new 'source' dimension
    stacked = xr.concat(aligned_tracks, dim="source")
    
    # Take the mean along the source dimension, ignoring NaNs
    fused = stacked.mean(dim="source", skipna=True)
    
    if print_report:
        print(report_nan_values(fused, "Fused track (mean)"))
    
    return fused


@log_to_attrs
def fuse_tracks_median(
    aligned_tracks: List[xr.DataArray],
    print_report: bool = False,
) -> xr.DataArray:
    """Fuse tracks by taking the median across all sources.

    Parameters
    ----------
    aligned_tracks : list of xarray.DataArray
        List of aligned position DataArrays.
    print_report : bool, optional
        Whether to print a report on the number of NaNs in the result, by default False.

    Returns
    -------
    xarray.DataArray
        Fused track with position values being the median across sources.

    Notes
    -----
    This function computes the median of all valid position values at each time point.
    If all sources have NaN at a particular time point, the result will also be NaN.
    This method is more robust to outliers than the mean method.
    """
    if not aligned_tracks:
        raise log_error(ValueError, "No tracks provided")

    # Stack all tracks along a new 'source' dimension
    stacked = xr.concat(aligned_tracks, dim="source")
    
    # Take the median along the source dimension, ignoring NaNs
    fused = stacked.median(dim="source", skipna=True)
    
    if print_report:
        print(report_nan_values(fused, "Fused track (median)"))
    
    return fused


@log_to_attrs
def fuse_tracks_weighted(
    aligned_tracks: List[xr.DataArray],
    weights: List[float] = None,
    confidence_arrays: List[xr.DataArray] = None,
    print_report: bool = False,
) -> xr.DataArray:
    """Fuse tracks using a weighted average.

    Parameters
    ----------
    aligned_tracks : list of xarray.DataArray
        List of aligned position DataArrays.
    weights : list of float, optional
        Static weights for each track source. Must sum to 1 if provided.
        If not provided and confidence_arrays is also None, equal weights are used.
    confidence_arrays : list of xarray.DataArray, optional
        Dynamic confidence values for each track. Must match the shape of aligned_tracks.
        If provided, these are used instead of static weights.
    print_report : bool, optional
        Whether to print a report on the number of NaNs in the result, by default False.

    Returns
    -------
    xarray.DataArray
        Fused track with position values weighted by the specified weights or confidence values.

    Notes
    -----
    This function computes a weighted average of position values. Weights can be either:
    - Static (one weight per source)
    - Dynamic (confidence value for each position at each time point)
    If both weights and confidence_arrays are provided, confidence_arrays takes precedence.
    """
    if not aligned_tracks:
        raise log_error(ValueError, "No tracks provided")
    
    n_tracks = len(aligned_tracks)
    
    # Check and prepare weights
    if weights is not None:
        if len(weights) != n_tracks:
            raise log_error(
                ValueError,
                f"Number of weights ({len(weights)}) does not match "
                f"number of tracks ({n_tracks})"
            )
        if abs(sum(weights) - 1.0) > 1e-10:
            raise log_error(
                ValueError,
                f"Weights must sum to 1, got sum={sum(weights)}"
            )
    else:
        # Equal weights if nothing is provided
        weights = [1.0 / n_tracks] * n_tracks
    
    # Use dynamic confidence arrays if provided
    if confidence_arrays is not None:
        if len(confidence_arrays) != n_tracks:
            raise log_error(
                ValueError,
                f"Number of confidence arrays ({len(confidence_arrays)}) does not match "
                f"number of tracks ({n_tracks})"
            )
        
        # Normalize confidence values per time point
        # Stack all confidence arrays along a 'source' dimension
        stacked_conf = xr.concat(confidence_arrays, dim="source")
        
        # Calculate sum of confidences at each time point
        sum_conf = stacked_conf.sum(dim="source")
        
        # Handle zeros by replacing with equal weights
        has_zeros = (sum_conf == 0)
        norm_conf = stacked_conf / sum_conf
        norm_conf = norm_conf.where(~has_zeros, 1.0 / n_tracks)
        
        # Apply confidence-weighted average
        stacked_pos = xr.concat(aligned_tracks, dim="source")
        weighted_pos = stacked_pos * norm_conf
        fused = weighted_pos.sum(dim="source", skipna=True)
    
    else:
        # Apply static weights
        weighted_tracks = [track * weight for track, weight in zip(aligned_tracks, weights)]
        
        # Stack and sum along a new 'source' dimension
        stacked = xr.concat(weighted_tracks, dim="source")
        
        # Calculate where all tracks are NaN
        all_nan = xr.concat([track.isnull() for track in aligned_tracks], dim="source").all(dim="source")
        
        # Sum along source dimension, set result to NaN where all sources are NaN
        fused = stacked.sum(dim="source", skipna=True).where(~all_nan)
    
    if print_report:
        print(report_nan_values(fused, "Fused track (weighted average)"))
    
    return fused


@log_to_attrs
def fuse_tracks_reliability(
    aligned_tracks: List[xr.DataArray],
    reliability_metrics: List[float] = None,
    window_size: int = 11,
    print_report: bool = False,
) -> xr.DataArray:
    """Fuse tracks by selecting the most reliable source at each time point.

    Parameters
    ----------
    aligned_tracks : list of xarray.DataArray
        List of aligned position DataArrays.
    reliability_metrics : list of float, optional
        Global reliability score for each source (higher is better).
        If not provided, NaN count is used as an inverse reliability metric.
    window_size : int, optional
        Window size for filtering the selection of sources, by default 11.
        Must be an odd number.
    print_report : bool, optional
        Whether to print a report on the number of NaNs in the result, by default False.

    Returns
    -------
    xarray.DataArray
        Fused track with position values taken from the most reliable source at each time.

    Notes
    -----
    This function selects values from the most reliable source at each time point,
    then applies a median filter to avoid rapid switching between sources, which
    could create unrealistic jumps in the trajectory.
    """
    if not aligned_tracks:
        raise log_error(ValueError, "No tracks provided")
    
    if window_size % 2 == 0:
        raise log_error(ValueError, "Window size must be an odd number")
    
    n_tracks = len(aligned_tracks)
    
    # Determine track reliability if not provided
    if reliability_metrics is None:
        # Count NaNs in each track (fewer NaNs = more reliable)
        nan_counts = [float(track.isnull().sum().values) for track in aligned_tracks]
        total_values = float(aligned_tracks[0].size)
        # Convert to a reliability score (inverse of NaN proportion)
        reliability_metrics = [1.0 - (count / total_values) for count in nan_counts]
    
    # Stack all tracks along a new 'source' dimension
    stacked = xr.concat(aligned_tracks, dim="source")
    
    # For each time point, create a selection array based on reliability and NaN status
    time_points = stacked.time.values
    selected_sources = np.zeros(len(time_points), dtype=int)
    
    # Loop through each time point
    for i, t in enumerate(time_points):
        values_at_t = [track.sel(time=t).values for track in aligned_tracks]
        is_nan = [np.isnan(val).any() for val in values_at_t]
        
        # If all sources have NaN, pick the most reliable one anyway
        if all(is_nan):
            selected_sources[i] = np.argmax(reliability_metrics)
        else:
            # Filter out NaN sources
            valid_indices = [idx for idx, nan_status in enumerate(is_nan) if not nan_status]
            valid_reliability = [reliability_metrics[idx] for idx in valid_indices]
            
            # Select the most reliable valid source
            best_valid_idx = valid_indices[np.argmax(valid_reliability)]
            selected_sources[i] = best_valid_idx
    
    # Apply median filter to smooth source selection and avoid rapid switching
    if window_size > 1 and len(time_points) > window_size:
        selected_sources = medfilt(selected_sources, window_size)
    
    # Create the fused track by selecting values from the chosen source at each time
    fused_data = np.zeros((len(time_points), stacked.sizes["space"]))
    
    for i, (t, source_idx) in enumerate(zip(time_points, selected_sources)):
        fused_data[i] = stacked.sel(time=t, source=source_idx).values
    
    # Create a new DataArray with the fused data
    fused = xr.DataArray(
        data=fused_data,
        dims=["time", "space"],
        coords={
            "time": time_points,
            "space": stacked.space.values
        }
    )
    
    if print_report:
        print(report_nan_values(fused, "Fused track (reliability-based)"))
    
    return fused


@log_to_attrs
def fuse_tracks_kalman(
    aligned_tracks: List[xr.DataArray],
    process_noise_scale: float = 0.01,
    measurement_noise_scales: List[float] = None,
    print_report: bool = False,
) -> xr.DataArray:
    """Fuse tracks using a Kalman filter.

    Parameters
    ----------
    aligned_tracks : list of xarray.DataArray
        List of aligned position DataArrays.
    process_noise_scale : float, optional
        Scale factor for the process noise covariance, by default 0.01.
    measurement_noise_scales : list of float, optional
        Scale factors for measurement noise for each source.
        Lower values indicate more reliable sources. Default is equal values.
    print_report : bool, optional
        Whether to print a report on the number of NaNs in the result, by default False.

    Returns
    -------
    xarray.DataArray
        Fused track with position values estimated by the Kalman filter.

    Notes
    -----
    This function implements a simple Kalman filter for track fusion. The filter:
    1. Models position and velocity in a state vector
    2. Predicts the next state based on constant velocity assumptions
    3. Updates the prediction using measurements from all available sources
    4. Handles missing measurements (NaNs) by skipping the update step
    
    The Kalman filter is particularly effective for trajectory smoothing and
    handling noisy measurements from multiple sources.
    """
    if not aligned_tracks:
        raise log_error(ValueError, "No tracks provided")
    
    n_tracks = len(aligned_tracks)
    
    # Set default measurement noise scales if not provided
    if measurement_noise_scales is None:
        measurement_noise_scales = [1.0] * n_tracks
    
    if len(measurement_noise_scales) != n_tracks:
        raise log_error(
            ValueError,
            f"Number of measurement noise scales ({len(measurement_noise_scales)}) "
            f"does not match number of tracks ({n_tracks})"
        )
    
    # Get the common time axis
    time_points = aligned_tracks[0].time.values
    n_timesteps = len(time_points)
    
    # Get the dimensionality of the space (2D or 3D)
    n_dims = len(aligned_tracks[0].space.values)
    
    # Initialize state vector [x, y, vx, vy] or [x, y, z, vx, vy, vz]
    state_dim = 2 * n_dims
    state = np.zeros(state_dim)
    
    # Initialize state covariance matrix
    state_cov = np.eye(state_dim)
    
    # Define transition matrix (constant velocity model)
    dt = 1.0  # Assuming unit time steps
    A = np.eye(state_dim)
    for i in range(n_dims):
        A[i, i + n_dims] = dt
    
    # Define process noise covariance
    Q = np.eye(state_dim) * process_noise_scale
    
    # Define measurement matrix (extracts position from state)
    H = np.zeros((n_dims, state_dim))
    for i in range(n_dims):
        H[i, i] = 1.0
    
    # Initialize storage for Kalman filter output
    kalman_output = np.zeros((n_timesteps, n_dims))
    
    # For the first time step, initialize with the average of available measurements
    first_measurements = []
    for track in aligned_tracks:
        pos = track.sel(time=time_points[0]).values
        if not np.isnan(pos).any():
            first_measurements.append(pos)
    
    if first_measurements:
        initial_pos = np.mean(first_measurements, axis=0)
        state[:n_dims] = initial_pos
        kalman_output[0] = initial_pos
    
    # Run Kalman filter
    for t in range(1, n_timesteps):
        # Prediction step
        state = A @ state
        state_cov = A @ state_cov @ A.T + Q
        
        # Update step - combine all available measurements
        measurements = []
        R_list = []  # Measurement noise covariances
        
        for i, track in enumerate(aligned_tracks):
            pos = track.sel(time=time_points[t]).values
            if not np.isnan(pos).any():
                measurements.append(pos)
                # Measurement noise covariance for this source
                R = np.eye(n_dims) * measurement_noise_scales[i]
                R_list.append(R)
        
        # Skip update if no measurements available
        if not measurements:
            kalman_output[t] = state[:n_dims]
            continue
        
        # Apply update for each measurement
        for z, R in zip(measurements, R_list):
            y = z - H @ state  # Measurement residual
            S = H @ state_cov @ H.T + R  # Residual covariance
            K = state_cov @ H.T @ np.linalg.inv(S)  # Kalman gain
            state = state + K @ y  # Updated state
            state_cov = (np.eye(state_dim) - K @ H) @ state_cov  # Updated covariance
        
        # Store the updated position
        kalman_output[t] = state[:n_dims]
    
    # Create a new DataArray with the Kalman filter output
    fused = xr.DataArray(
        data=kalman_output,
        dims=["time", "space"],
        coords={
            "time": time_points,
            "space": aligned_tracks[0].space.values
        }
    )
    
    if print_report:
        print(report_nan_values(fused, "Fused track (Kalman filter)"))
    
    return fused


@log_to_attrs
def fuse_tracks(
    datasets: List[xr.Dataset],
    method: Union[str, FusionMethod] = "kalman",
    keypoint: str = "centroid",
    interpolate_gaps: bool = True,
    max_gap: int = 5,
    weights: List[float] = None,
    confidence_arrays: List[xr.DataArray] = None,
    reliability_metrics: List[float] = None,
    window_size: int = 11,
    process_noise_scale: float = 0.01,
    measurement_noise_scales: List[float] = None,
    print_report: bool = False,
) -> xr.DataArray:
    """Fuse tracks from multiple datasets using the specified method.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of datasets containing position data to fuse.
    method : str or FusionMethod, optional
        Track fusion method to use, by default "kalman". Options are:
        - "mean": Average position across all sources
        - "median": Median position across all sources (robust to outliers)
        - "weighted": Weighted average using static weights or confidence values
        - "reliability": Select most reliable source at each time point
        - "kalman": Apply Kalman filter to estimate the optimal trajectory
    keypoint : str, optional
        The keypoint to extract from each dataset, by default "centroid".
    interpolate_gaps : bool, optional
        Whether to interpolate missing values after alignment, by default True.
    max_gap : int, optional
        Maximum size of gap to interpolate, by default 5.
    weights : list of float, optional
        Static weights for each track source (used with "weighted" method).
    confidence_arrays : list of xarray.DataArray, optional
        Dynamic confidence values for each track (used with "weighted" method).
    reliability_metrics : list of float, optional
        Global reliability score for each source (used with "reliability" method).
    window_size : int, optional
        Window size for filtering source selection (used with "reliability" method).
    process_noise_scale : float, optional
        Scale factor for process noise (used with "kalman" method).
    measurement_noise_scales : list of float, optional
        Scale factors for measurement noise (used with "kalman" method).
    print_report : bool, optional
        Whether to print a report on the number of NaNs in the result, by default False.

    Returns
    -------
    xarray.DataArray
        Fused track with position values determined by the specified fusion method.

    Raises
    ------
    ValueError
        If an unsupported fusion method is specified or parameters are invalid.

    Notes
    -----
    This function acts as a high-level interface to various track fusion methods,
    automatically handling dataset alignment and applying the selected fusion algorithm.
    """
    # Convert string method to enum if needed
    if isinstance(method, str):
        method_map = {
            "mean": FusionMethod.MEAN,
            "median": FusionMethod.MEDIAN,
            "weighted": FusionMethod.WEIGHTED,
            "reliability": FusionMethod.RELIABILITY_BASED,
            "kalman": FusionMethod.KALMAN,
        }
        
        if method.lower() not in method_map:
            valid_methods = list(method_map.keys())
            raise log_error(
                ValueError,
                f"Unsupported fusion method: {method}. "
                f"Valid methods are: {valid_methods}"
            )
        
        method = method_map[method.lower()]
    
    # Align datasets
    aligned_tracks = align_datasets(
        datasets=datasets,
        keypoint=keypoint,
        interpolate=interpolate_gaps,
        max_gap=max_gap,
    )
    
    # Apply fusion method
    if method == FusionMethod.MEAN:
        return fuse_tracks_mean(
            aligned_tracks=aligned_tracks,
            print_report=print_report,
        )
    
    elif method == FusionMethod.MEDIAN:
        return fuse_tracks_median(
            aligned_tracks=aligned_tracks,
            print_report=print_report,
        )
    
    elif method == FusionMethod.WEIGHTED:
        return fuse_tracks_weighted(
            aligned_tracks=aligned_tracks,
            weights=weights,
            confidence_arrays=confidence_arrays,
            print_report=print_report,
        )
    
    elif method == FusionMethod.RELIABILITY_BASED:
        return fuse_tracks_reliability(
            aligned_tracks=aligned_tracks,
            reliability_metrics=reliability_metrics,
            window_size=window_size,
            print_report=print_report,
        )
    
    elif method == FusionMethod.KALMAN:
        return fuse_tracks_kalman(
            aligned_tracks=aligned_tracks,
            process_noise_scale=process_noise_scale,
            measurement_noise_scales=measurement_noise_scales,
            print_report=print_report,
        )
    
    else:
        raise log_error(
            ValueError,
            f"Unsupported fusion method: {method}"
        ) 