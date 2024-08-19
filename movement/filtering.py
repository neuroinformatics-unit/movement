"""Filter and interpolate tracks in ``movement`` datasets."""

import xarray as xr
from scipy import signal

from movement.utils.logging import log_error, log_to_attrs
from movement.utils.reports import report_nan_values


@log_to_attrs
def filter_by_confidence(
    data: xr.DataArray,
    confidence: xr.DataArray,
    threshold: float = 0.6,
    print_report: bool = True,
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
        before and after filtering. Default is ``True``.

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
    method: str = "linear",
    max_gap: int | None = None,
    print_report: bool = True,
) -> xr.DataArray:
    """Fill in NaN values by interpolating over the ``time`` dimension.

    This method uses :meth:`xarray.DataArray.interpolate_na` under the
    hood and passes the ``method`` and ``max_gap`` parameters to it.
    See the xarray documentation for more details on these parameters.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be interpolated.
    method : str
        String indicating which method to use for interpolation.
        Default is ``linear``.
    max_gap : int, optional
        Maximum size of gap, a continuous sequence of missing observations
        (represented as NaNs), to fill.
        The default value is ``None`` (no limit).
        Gap size is defined as the number of consecutive NaNs.
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after interpolation. Default is ``True``.

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
        fill_value="extrapolate",
    )
    if print_report:
        print(report_nan_values(data, "input"))
        print(report_nan_values(data_interpolated, "output"))
    return data_interpolated


@log_to_attrs
def median_filter(
    data: xr.DataArray,
    window: int,
    min_periods: int | None = None,
    print_report: bool = True,
) -> xr.DataArray:
    """Smooth data by applying a median filter over time.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be smoothed.
    window : int
        The size of the smoothing window, representing the fixed number
        of observations used for each window.
    min_periods : int
        Minimum number of observations in the window required to have
        a value (otherwise result is NaN). The default, None, is
        equivalent to setting ``min_periods`` equal to the size of the window.
        This argument is directly  passed to the ``min_periods`` parameter of
        :meth:`xarray.DataArray.rolling`.
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after smoothing. Default is ``True``.

    Returns
    -------
    xarray.DataArray
        The data smoothed using a median filter with the provided parameters.

    Notes
    -----
    By default, whenever one or more NaNs are present in the smoothing window,
    a NaN is returned to the output array. As a result, any
    stretch of NaNs present in the input data will be propagated
    proportionally to the size of the window  (specifically, by
    ``floor(window/2)``). To control this behaviour, the
    ``min_periods`` option can be used to specify the minimum number of
    non-NaN values required in the window to compute a result. For example,
    setting ``min_periods=1`` will result in the filter returning NaNs
    only when all values in the window are NaN, since 1 non-NaN value
    is sufficient to compute the median.

    """
    half_window = window // 2
    data_smoothed = (
        data.pad(  # Pad the edges to avoid NaNs
            time=half_window, mode="reflect"
        )
        .rolling(  # Take rolling windows across time
            time=window, center=True, min_periods=min_periods
        )
        .median(  # Compute the median of each window
            skipna=True
        )
        .isel(  # Remove the padded edges
            time=slice(half_window, -half_window)
        )
    )
    if print_report:
        print(report_nan_values(data, "input"))
        print(report_nan_values(data_smoothed, "output"))
    return data_smoothed


@log_to_attrs
def savgol_filter(
    data: xr.DataArray,
    window: int,
    polyorder: int = 2,
    print_report: bool = True,
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
        before and after smoothing. Default is ``True``.
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
    :func:`movement.filtering.median_filter`, there is no ``min_periods``
    option to control this behaviour.

    """
    if "axis" in kwargs:
        raise log_error(
            ValueError, "The 'axis' argument may not be overridden."
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
