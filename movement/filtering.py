"""Filter and interpolate tracks in ``movement`` datasets."""

from typing import Any, Literal

import xarray as xr
from scipy import signal
from xarray.core.types import InterpOptions

from movement.utils.logging import log_to_attrs, logger
from movement.utils.reports import report_nan_values


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
    data
        The input data to be filtered.
    confidence
        The data array containing confidence scores to filter by.
    threshold
        The confidence threshold below which datapoints are filtered.
        A default value of ``0.6`` is used. See notes for more information.
    print_report
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
    data
        The input data to be interpolated.
    method
        String indicating which method to use for interpolation.
        Default is ``linear``.
    max_gap
        Maximum size of gap, a continuous sequence of missing observations
        (represented as NaNs), to fill.
        The default value is ``None`` (no limit).
        Gap size is defined as the number of consecutive NaNs
        (see Notes for more information).
    print_report
        Whether to print a report on the number of NaNs in the dataset
        before and after interpolation. Default is ``False``.
    **kwargs
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
    data
        The input data array.
    window
        The size of the rolling window, representing the fixed number
        of observations used for each window.
    statistic
        Which statistic to compute over the rolling window.
        Options are ``median``, ``mean``, ``max``, and ``min``.
        The default is ``median``.
    min_periods
        Minimum number of observations in the window required to have
        a value (otherwise result is NaN). The default, None, is
        equivalent to setting ``min_periods`` equal to the size of the window.
        This argument is directly passed to the ``min_periods`` parameter of
        :meth:`xarray.DataArray.rolling`.
    print_report
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
    data
        The input data to be smoothed.
    window
        The size of the smoothing window, representing the fixed number
        of observations used for each window.
    polyorder
        The order of the polynomial used to fit the samples. Must be
        less than ``window``. By default, a ``polyorder`` of
        2 is used.
    print_report
        Whether to print a report on the number of NaNs in the dataset
        before and after smoothing. Default is ``False``.
    **kwargs
        Additional keyword arguments are passed to
        :func:`scipy.signal.savgol_filter`.
        Note that the ``axis`` keyword argument may not be overridden,
        as the filter is always applied over the ``time`` dimension.


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

    The function raises a ``ValueError`` if NaNs are found within the signal's
    edge windows. To avoid this, fill any edge NaNs before filtering, or switch
    from the default ``mode='interp'`` to an alternative edge handling
    mode (e.g., ``mode='nearest'`` or ``mode='mirror'``).

    """
    if "axis" in kwargs:
        raise logger.error(
            ValueError("The 'axis' argument may not be overridden.")
        )
    time_axis = data.get_axis_num("time")
    data_smoothed = data.copy()
    try:
        data_smoothed.values = signal.savgol_filter(
            data,
            window,
            polyorder,
            axis=time_axis,
            **kwargs,
        )
    except ValueError as e:
        if "array must not contain infs or NaNs" in str(e):
            raise logger.error(
                ValueError(
                    "mode='interp' does not support NaNs in edge windows "
                    "with SciPy >= 1.17; use mode='nearest'/'mirror' or "
                    "fill edge NaNs before filtering."
                )
            ) from e
        raise  # Re-raise any other ValueError unchanged
    if print_report:
        print(report_nan_values(data, "input"))
        print(report_nan_values(data_smoothed, "output"))
    return data_smoothed


@log_to_attrs
def filter_valid_regions(
    data: xr.DataArray,
    max_nan_fraction: float = 0.0,
    min_length: int = 2,
) -> xr.DataArray:
    """Select the longest contiguous region with an acceptable NaN fraction.

    Scans the ``time`` dimension and finds the longest contiguous stretch of
    frames in which the fraction of NaN values (computed across all non-time
    dimensions) does not exceed ``max_nan_fraction``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to filter. Must contain a ``time`` dimension.
    max_nan_fraction : float
        The maximum fraction of NaN values allowed at a single time step,
        computed across all non-time dimensions (e.g. space, keypoints,
        individuals). Must be between 0 and 1 (inclusive).
        Default is ``0.0`` (no NaNs allowed at any time step).
    min_length : int
        The minimum number of consecutive time frames required for a region
        to be kept. Default is ``2``.

    Returns
    -------
    xarray.DataArray
        The input data sliced to the longest contiguous time region that
        satisfies ``max_nan_fraction``. The ``time`` coordinates are
        preserved from the original array.

    Raises
    ------
    ValueError
        If ``max_nan_fraction`` is not between 0 and 1.
    ValueError
        If ``min_length`` is less than 1.
    ValueError
        If no valid region of at least ``min_length`` frames exists.

    Examples
    --------
    Keep only the longest contiguous stretch with no NaNs:

    >>> position_clean = filter_valid_regions(position)

    Allow up to 20 % NaNs per time frame, requiring at least 100 frames:

    >>> position_clean = filter_valid_regions(
    ...     position, max_nan_fraction=0.2, min_length=100
    ... )

    See Also
    --------
    movement.filtering.interpolate_over_time :
        Fill NaN gaps by interpolation before or after filtering.

    """
    if not (0.0 <= max_nan_fraction <= 1.0):
        raise logger.error(
            ValueError(
                f"'max_nan_fraction' must be between 0 and 1, "
                f"got {max_nan_fraction!r}."
            )
        )
    if min_length < 1:
        raise logger.error(
            ValueError(f"'min_length' must be at least 1, got {min_length!r}.")
        )

    # Compute NaN fraction at each time step across all non-time dimensions.
    other_dims = [dim for dim in data.dims if dim != "time"]
    if other_dims:
        n_non_time = data.size // data.sizes["time"]
        nan_fraction = data.isnull().sum(dim=other_dims) / n_non_time
    else:
        nan_fraction = data.isnull().astype(float)

    valid_mask = (nan_fraction <= max_nan_fraction).values  # 1-D bool array

    # Find the longest contiguous run of True values.
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0

    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            if current_length == 0:
                current_start = i
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
        else:
            current_length = 0

    if best_length < min_length:
        raise logger.error(
            ValueError(
                f"No valid region of at least {min_length} frame(s) found. "
                f"The longest valid region has {best_length} frame(s). "
                f"Try increasing 'max_nan_fraction' or decreasing "
                f"'min_length'."
            )
        )

    return data.isel(time=slice(best_start, best_start + best_length))
