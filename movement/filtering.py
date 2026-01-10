"""Filter and interpolate tracks in ``movement`` datasets."""

from typing import Any, Literal

import numpy as np
import xarray as xr
from scipy import signal
from xarray.core.types import InterpOptions

from movement.utils.logging import log_to_attrs, logger
from movement.utils.reports import report_nan_values
from movement.validators.arrays import validate_dims_coords


@log_to_attrs
def filter_by_confidence(
    data: xr.DataArray,
    confidence: xr.DataArray,
    threshold: float = 0.6,
    print_report: bool = False,
) -> xr.DataArray:
    """Drop data points below a certain confidence threshold.

    Data points with an associated confidence value below the threshold are
    converted to NaN.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be filtered.
    confidence : xarray.DataArray
        The data array containing confidence scores to filter by.
    threshold : float
        The confidence threshold below which datapoints are filtered.
        A default value of ``0.6`` is used. See notes for more information.
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after filtering. Default is ``False``.

    Returns
    -------
    xarray.DataArray
        The data where points with a confidence value below the
        user-defined threshold have been converted to NaNs.

    Notes
    -----
    For the poses dataset case, note that the point-wise confidence values
    reported by various pose estimation frameworks are not standardised, and
    the range of values can vary. For example, DeepLabCut reports a likelihood
    value between 0 and 1, whereas the point confidence reported by SLEAP can
    range above 1. Therefore, the default threshold value will not be
    appropriate for all datasets and does not have the same meaning across
    pose estimation frameworks. We advise users to inspect the confidence
    values in their dataset and adjust the threshold accordingly.

    """
    data_filtered = data.where(confidence >= threshold)
    if print_report:
        print(report_nan_values(data, "input"))
        print(report_nan_values(data_filtered, "output"))
    return data_filtered


@log_to_attrs
def interpolate_over_time(
    data: xr.DataArray,
    method: InterpOptions = "linear",
    max_gap: int | None = None,
    print_report: bool = False,
    **kwargs: Any,
) -> xr.DataArray:
    """Fill in NaN values by interpolating over the ``time`` dimension.

    This function calls :meth:`xarray.DataArray.interpolate_na` and can pass
    additional keyword arguments to it, depending on the chosen ``method``.
    See the xarray documentation for more details.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be interpolated.
    method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", \
        "polynomial", "barycentric", "krogh", "pchip", "spline", "akima"}
        String indicating which method to use for interpolation.
        Default is ``linear``.
    max_gap : int, optional
        Maximum size of gap, a continuous sequence of missing observations
        (represented as NaNs), to fill.
        The default value is ``None`` (no limit).
        Gap size is defined as the number of consecutive NaNs
        (see Notes for more information).
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after interpolation. Default is ``False``.
    **kwargs : Any
        Any ``**kwargs`` accepted by :meth:`xarray.DataArray.interpolate_na`,
        which in turn passes them verbatim to the underlying
        interpolation methods.

    Returns
    -------
    xarray.DataArray
        The data where NaN values have been interpolated over
        using the parameters provided.

    Notes
    -----
    The ``max_gap`` parameter differs slightly from that in
    :meth:`xarray.DataArray.interpolate_na`, in which the gap size
    is defined as the difference between the ``time`` coordinate values
    at the first data point after a gap and the last value before a gap.

    """
    data_interpolated = data.interpolate_na(
        dim="time",
        method=method,
        use_coordinate=False,
        max_gap=max_gap + 1 if max_gap is not None else None,
        **kwargs,
    )
    if print_report:
        print(report_nan_values(data, "input"))
        print(report_nan_values(data_interpolated, "output"))
    return data_interpolated


@log_to_attrs
def rolling_filter(
    data: xr.DataArray,
    window: int,
    statistic: Literal["median", "mean", "max", "min"] = "median",
    min_periods: int | None = None,
    print_report: bool = False,
) -> xr.DataArray:
    """Apply a rolling window filter across time.

    This function uses :meth:`xarray.DataArray.rolling` to apply a rolling
    window filter across the ``time`` dimension of the input data and computes
    the specified ``statistic`` over each window.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array.
    window : int
        The size of the rolling window, representing the fixed number
        of observations used for each window.
    statistic : str
        Which statistic to compute over the rolling window.
        Options are ``median``, ``mean``, ``max``, and ``min``.
        The default is ``median``.
    min_periods : int
        Minimum number of observations in the window required to have
        a value (otherwise result is NaN). The default, None, is
        equivalent to setting ``min_periods`` equal to the size of the window.
        This argument is directly passed to the ``min_periods`` parameter of
        :meth:`xarray.DataArray.rolling`.
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after smoothing. Default is ``False``.

    Returns
    -------
    xarray.DataArray
        The filtered data array.

    Notes
    -----
    By default, whenever one or more NaNs are present in the window,
    a NaN is returned to the output array. As a result, any
    stretch of NaNs present in the input data will be propagated
    proportionally to the size of the window (specifically, by
    ``floor(window/2)``). To control this behaviour, the
    ``min_periods`` option can be used to specify the minimum number of
    non-NaN values required in the window to compute a result. For example,
    setting ``min_periods=1`` will result in the filter returning NaNs
    only when all values in the window are NaN, since 1 non-NaN value
    is sufficient to compute the result.

    """
    half_window = window // 2
    data_windows = data.pad(  # Pad the edges to avoid NaNs
        time=half_window, mode="reflect"
    ).rolling(  # Take rolling windows across time
        time=window, center=True, min_periods=min_periods
    )

    # Compute the statistic over each window
    allowed_statistics = ["mean", "median", "max", "min"]
    if statistic not in allowed_statistics:
        raise logger.error(
            ValueError(
                f"Invalid statistic '{statistic}'. "
                f"Must be one of {allowed_statistics}."
            )
        )

    data_rolled = getattr(data_windows, statistic)(skipna=True)

    # Remove the padded edges
    data_rolled = data_rolled.isel(time=slice(half_window, -half_window))

    # Optional: Print NaN report
    if print_report:
        print(report_nan_values(data, "input"))
        print(report_nan_values(data_rolled, "output"))

    return data_rolled


@log_to_attrs
def savgol_filter(
    data: xr.DataArray,
    window: int,
    polyorder: int = 2,
    print_report: bool = False,
    **kwargs,
) -> xr.DataArray:
    """Smooth data by applying a Savitzky-Golay filter over time.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be smoothed.
    window : int
        The size of the smoothing window, representing the fixed number
        of observations used for each window.
    polyorder : int
        The order of the polynomial used to fit the samples. Must be
        less than ``window``. By default, a ``polyorder`` of
        2 is used.
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after smoothing. Default is ``False``.
    **kwargs : dict
        Additional keyword arguments are passed to
        :func:`scipy.signal.savgol_filter`.
        Note that the ``axis`` keyword argument may not be overridden.


    Returns
    -------
    xarray.DataArray
        The data smoothed using a Savitzky-Golay filter with the
        provided parameters.

    Notes
    -----
    Uses the :func:`scipy.signal.savgol_filter` function to apply a
    Savitzky-Golay filter to the input data.
    See the SciPy documentation for more information on that function.
    Whenever one or more NaNs are present in a smoothing window of the
    input data, a NaN is returned to the output array. As a result, any
    stretch of NaNs present in the input data will be propagated
    proportionally to the size of the window (specifically, by
    ``floor(window/2)``). Note that, unlike
    :func:`movement.filtering.rolling_filter`, there is no ``min_periods``
    option to control this behaviour.

    """
    if "axis" in kwargs:
        raise logger.error(
            ValueError("The 'axis' argument may not be overridden.")
        )
    data_smoothed = data.copy()
    data_smoothed.values = signal.savgol_filter(
        data,
        window,
        polyorder,
        axis=0,
        **kwargs,
    )
    if print_report:
        print(report_nan_values(data, "input"))
        print(report_nan_values(data_smoothed, "output"))
    return data_smoothed


def _get_transition_matrix(dt: float, n_space: int) -> np.ndarray:
    """Construct the transition matrix for constant acceleration model.

    The state vector contains [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
    for 2D space, or [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z]
    for 3D space.

    Parameters
    ----------
    dt : float
        Time step between measurements.
    n_space : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    numpy.ndarray
        Transition matrix F of shape (n_state, n_state) where
        n_state = 3 * n_space.
    """
    if n_space not in [2, 3]:
        raise logger.error(
            ValueError(f"n_space must be 2 or 3, got {n_space}.")
        )

    n_state = 3 * n_space
    F = np.zeros((n_state, n_state))

    # For each spatial dimension, set up the constant acceleration model:
    # pos = old_pos + vel*dt + 0.5*acc*dt^2
    # vel = old_vel + acc*dt
    # acc = old_acc (constant)
    for i in range(n_space):
        base_idx = i
        vel_idx = n_space + i
        acc_idx = 2 * n_space + i

        # Position update
        F[base_idx, base_idx] = 1.0  # pos = pos
        F[base_idx, vel_idx] = dt  # pos += vel*dt
        F[base_idx, acc_idx] = 0.5 * dt**2  # pos += 0.5*acc*dt^2

        # Velocity update
        F[vel_idx, vel_idx] = 1.0  # vel = vel
        F[vel_idx, acc_idx] = dt  # vel += acc*dt

        # Acceleration update (constant)
        F[acc_idx, acc_idx] = 1.0  # acc = acc

    return F


def _get_measurement_matrix(n_space: int) -> np.ndarray:
    """Construct the measurement matrix H.

    We only measure position (not velocity or acceleration directly).

    Parameters
    ----------
    n_space : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    numpy.ndarray
        Measurement matrix H of shape (n_space, n_state) where
        n_state = 3 * n_space.
    """
    if n_space not in [2, 3]:
        raise logger.error(
            ValueError(f"n_space must be 2 or 3, got {n_space}.")
        )

    n_state = 3 * n_space
    H = np.zeros((n_space, n_state))

    # We only measure position (first n_space elements of state vector)
    for i in range(n_space):
        H[i, i] = 1.0

    return H


def _kalman_filter_1d(
    measurements: np.ndarray,
    dt: float,
    process_noise: float = 0.01,
    measurement_noise: float = 1.0,
    initial_covariance: float = 0.1,
) -> np.ndarray:
    """Apply Kalman filter to a single 1D track (time series).

    This function processes a single trajectory along the time dimension.
    The state vector includes position, velocity, and acceleration for each
    spatial dimension.

    Parameters
    ----------
    measurements : numpy.ndarray
        Measurement array of shape (n_time, n_space) containing position
        measurements. Missing values should be NaN.
    dt : float
        Time step between measurements. If measurements are equally spaced
        in time, this should be the sampling interval. If not, this should
        be the average time step.
    process_noise : float
        Process noise covariance (Q matrix scaling factor). Larger values
        indicate more uncertainty in the dynamic model. Default is 0.01.
    measurement_noise : float
        Measurement noise covariance (R matrix scaling factor). Larger values
        indicate noisier measurements. Default is 1.0.
    initial_covariance : float
        Initial state covariance scaling factor. Default is 0.1.

    Returns
    -------
    numpy.ndarray
        Smoothed state estimates of shape (n_time, n_state) where
        n_state = 3 * n_space. The state vector contains [pos, vel, acc]
        for each spatial dimension.

    Notes
    -----
    This implementation uses a constant acceleration model. When measurements
    are missing (NaN), the filter predicts forward without updating, allowing
    it to bridge gaps in the data.

    """
    n_time, n_space = measurements.shape

    if n_space not in [2, 3]:
        raise ValueError(f"n_space must be 2 or 3, got {n_space}.")

    n_state = 3 * n_space

    # Initialize state: [pos, vel, acc] for each spatial dimension
    # If first measurement is available, use it; otherwise start at zero
    state = np.zeros(n_state)
    if not np.isnan(measurements[0]).any():
        state[:n_space] = measurements[0]

    # Initialize covariance
    covariance = np.eye(n_state) * initial_covariance

    # Get transition and measurement matrices
    F = _get_transition_matrix(dt, n_space)
    H = _get_measurement_matrix(n_space)

    # Noise matrices
    Q = np.eye(n_state) * process_noise  # Process noise
    R = np.eye(n_space) * measurement_noise  # Measurement noise

    # Storage for smoothed states
    smoothed_states = np.zeros((n_time, n_state))

    for t in range(n_time):
        # --- PREDICT STEP ---
        state_pred = F @ state
        cov_pred = F @ covariance @ F.T + Q

        # --- UPDATE STEP ---
        z = measurements[t]  # Current measurement

        # Handle missing measurements (NaN)
        if np.isnan(z).any():
            # If measurement is missing, use prediction only
            state = state_pred
            covariance = cov_pred
        else:
            # Standard Kalman update
            y = z - (H @ state_pred)  # Innovation (residual)
            S = H @ cov_pred @ H.T + R  # Innovation covariance
            K = cov_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
            state = state_pred + (K @ y)  # Updated state
            covariance = (np.eye(n_state) - K @ H) @ cov_pred  # Updated covariance

        smoothed_states[t] = state

    return smoothed_states


@log_to_attrs
def kalman_filter(
    data: xr.DataArray,
    dt: float | None = None,
    process_noise: float = 0.01,
    measurement_noise: float = 1.0,
    output: Literal["position", "velocity", "acceleration", "all"] = "position",
    print_report: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Smooth position data using a Kalman filter.

    This function applies a Kalman filter with a constant acceleration model
    to smooth position measurements over time. The filter estimates position,
    velocity, and acceleration while accounting for measurement noise and
    process uncertainty.

    Parameters
    ----------
    data : xarray.DataArray
        Input position data with dimensions including ``time`` and ``space``.
        Missing values should be NaN.
    dt : float, optional
        Time step between measurements. If ``None``, the function attempts to
        infer it from the ``time`` coordinate. If the time coordinate is in
        frames, ``dt=1.0`` is used. If the dataset has an ``fps`` attribute,
        ``dt = 1.0 / fps`` is used. Default is ``None``.
    process_noise : float
        Process noise covariance scaling factor. Larger values indicate more
        uncertainty in the dynamic model, leading to more responsive filtering
        but potentially less smoothing. Default is 0.01.
    measurement_noise : float
        Measurement noise covariance scaling factor. Larger values indicate
        noisier measurements, causing the filter to trust the model more than
        the measurements. Default is 1.0.
    output : {"position", "velocity", "acceleration", "all"}
        Which outputs to return. If ``"all"``, returns a Dataset with
        ``position``, ``velocity``, and ``acceleration`` DataArrays.
        Default is ``"position"``.
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after filtering. Default is ``False``.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        If ``output`` is ``"position"``, ``"velocity"``, or ``"acceleration"``,
        returns a DataArray with the smoothed values.
        If ``output`` is ``"all"``, returns a Dataset containing all three
        DataArrays with dimensions matching the input (except ``time`` is
        preserved).

    Notes
    -----
    The Kalman filter uses a constant acceleration dynamic model. When
    measurements are missing (NaN), the filter predicts forward based on the
    current velocity and acceleration estimates, allowing it to bridge gaps
    in the data.

    The filter parameters (``process_noise`` and ``measurement_noise``) may
    need to be tuned for your specific use case:
    - For noisier measurements, increase ``measurement_noise``.
    - For more variable motion, increase ``process_noise``.
    - For smoother output, decrease ``process_noise`` and/or increase
      ``measurement_noise``.

    Examples
    --------
    >>> import movement
    >>> from movement.filtering import kalman_filter
    >>> ds = movement.sample_data.fetch_dataset("DLC_single-wasp.predictions.h5")
    >>> # Smooth position data
    >>> position_smooth = kalman_filter(
    ...     ds.position,
    ...     process_noise=0.01,
    ...     measurement_noise=1.0
    ... )
    >>> # Get all outputs (position, velocity, acceleration)
    >>> results = kalman_filter(ds.position, output="all")
    >>> ds["position"] = results.position
    >>> ds["velocity"] = results.velocity
    >>> ds["acceleration"] = results.acceleration

    References
    ----------
    For an intuitive introduction to Kalman filters, see
    `KalmanFilter.net <https://kalmanfilter.net/>`_.

    """
    # Validate input dimensions
    validate_dims_coords(data, {"time": [], "space": []})

    # Determine time step
    if dt is None:
        if "fps" in data.attrs:
            dt = 1.0 / data.attrs["fps"]
        else:
            # Try to infer from time coordinate
            time_coords = data.coords["time"].values
            if len(time_coords) > 1:
                # Use median time step for robustness
                dt = float(np.median(np.diff(time_coords)))
            else:
                dt = 1.0
                logger.warning(
                    "Could not infer time step from data. Using dt=1.0. "
                    "Consider providing dt explicitly."
                )

    # Get spatial dimensions
    n_space = data.sizes["space"]
    if n_space not in [2, 3]:
        raise logger.error(
            ValueError(
                f"Kalman filter currently supports 2D or 3D space only. "
                f"Got {n_space} spatial dimensions."
            )
        )

    # Optional: Print NaN report
    if print_report:
        print(report_nan_values(data, "input"))

    # Define wrapper functions for extracting different outputs
    # Capture n_space in closure
    n_space_captured = n_space

    def kalman_filter_position(measurements, dt, process_noise, measurement_noise):
        """Apply Kalman filter and return position only."""
        states = _kalman_filter_1d(
            measurements,
            dt,
            process_noise,
            measurement_noise,
        )
        return states[:, :n_space_captured]

    def kalman_filter_velocity(measurements, dt, process_noise, measurement_noise):
        """Apply Kalman filter and return velocity only."""
        states = _kalman_filter_1d(
            measurements,
            dt,
            process_noise,
            measurement_noise,
        )
        return states[:, n_space_captured : 2 * n_space_captured]

    def kalman_filter_acceleration(
        measurements, dt, process_noise, measurement_noise
    ):
        """Apply Kalman filter and return acceleration only."""
        states = _kalman_filter_1d(
            measurements,
            dt,
            process_noise,
            measurement_noise,
        )
        return states[:, 2 * n_space_captured : 3 * n_space_captured]

    def kalman_filter_all(measurements, dt, process_noise, measurement_noise):
        """Apply Kalman filter and return all states (for Dataset output)."""
        return _kalman_filter_1d(
            measurements,
            dt,
            process_noise,
            measurement_noise,
        )

    # Apply Kalman filter based on desired output
    filter_kwargs = {
        "dt": dt,
        "process_noise": process_noise,
        "measurement_noise": measurement_noise,
    }

    # Preserve input dimension order
    input_dims_order = list(data.dims)

    if output == "position":
        filtered = xr.apply_ufunc(
            kalman_filter_position,
            data,
            input_core_dims=[["time", "space"]],
            output_core_dims=[["time", "space"]],
            vectorize=True,
            dask="allowed",
            kwargs=filter_kwargs,
        )
        # Restore dimension order to match input
        filtered = filtered.transpose(*input_dims_order)
        filtered.name = "position"
        result = filtered
    elif output == "velocity":
        filtered = xr.apply_ufunc(
            kalman_filter_velocity,
            data,
            input_core_dims=[["time", "space"]],
            output_core_dims=[["time", "space"]],
            vectorize=True,
            dask="allowed",
            kwargs=filter_kwargs,
        )
        # Restore dimension order to match input
        filtered = filtered.transpose(*input_dims_order)
        filtered.name = "velocity"
        result = filtered
    elif output == "acceleration":
        filtered = xr.apply_ufunc(
            kalman_filter_acceleration,
            data,
            input_core_dims=[["time", "space"]],
            output_core_dims=[["time", "space"]],
            vectorize=True,
            dask="allowed",
            kwargs=filter_kwargs,
        )
        # Restore dimension order to match input
        filtered = filtered.transpose(*input_dims_order)
        filtered.name = "acceleration"
        result = filtered
    elif output == "all":
        # For "all" output, we need to get all states and split them
        filtered_states = xr.apply_ufunc(
            kalman_filter_all,
            data,
            input_core_dims=[["time", "space"]],
            output_core_dims=[["time", "state"]],
            vectorize=True,
            dask="allowed",
            kwargs=filter_kwargs,
        )

        # Extract position, velocity, acceleration from state vector
        # State vector structure: [pos_0, pos_1, ..., vel_0, vel_1, ..., acc_0, acc_1, ...]
        position = filtered_states.isel(state=slice(0, n_space))
        position = position.rename({"state": "space"})
        position.name = "position"

        velocity = filtered_states.isel(state=slice(n_space, 2 * n_space))
        velocity = velocity.rename({"state": "space"})
        velocity.name = "velocity"

        acceleration = filtered_states.isel(state=slice(2 * n_space, 3 * n_space))
        acceleration = acceleration.rename({"state": "space"})
        acceleration.name = "acceleration"

        # Ensure correct dimension order: restore original order from input
        dims_order = list(data.dims)
        # Transpose to match input dimension order
        position = position.transpose(*dims_order)
        velocity = velocity.transpose(*dims_order)
        acceleration = acceleration.transpose(*dims_order)

        # Copy coordinates to ensure consistency
        for var in [position, velocity, acceleration]:
            for dim in dims_order:
                if dim in data.coords:
                    var.coords[dim] = data.coords[dim]

        result = xr.Dataset(
            {
                "position": position,
                "velocity": velocity,
                "acceleration": acceleration,
            }
        )
    else:
        raise logger.error(
            ValueError(
                f"Invalid output '{output}'. "
                f"Must be one of 'position', 'velocity', 'acceleration', 'all'."
            )
        )

    # Optional: Print NaN report
    if print_report:
        if isinstance(result, xr.Dataset):
            print(report_nan_values(result.position, "output (position)"))
        else:
            print(report_nan_values(result, "output"))

    return result
