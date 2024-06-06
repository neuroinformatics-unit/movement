"""Filter and interpolate  pose tracks in ``movement`` datasets."""

import logging
from datetime import datetime
from functools import wraps

import xarray as xr
from scipy import signal

from movement.utils.logging import log_error


def log_to_attrs(func):
    """Log the operation performed by the wrapped function.

    This decorator appends log entries to the dataset's ``log``
    attribute. The wrapped function must accept an ``xarray.Dataset``
    as its first argument and return an ``xarray.Dataset``.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        log_entry = {
            "operation": func.__name__,
            "datetime": str(datetime.now()),
            **{f"arg_{i}": arg for i, arg in enumerate(args[1:], start=1)},
            **kwargs,
        }

        # Append the log entry to the result's attributes
        if result is not None and hasattr(result, "attrs"):
            if "log" not in result.attrs:
                result.attrs["log"] = []
            result.attrs["log"].append(log_entry)

        return result

    return wrapper


def generate_nan_report(
    data: xr.DataArray, keypoint: str, individual: str | None = None
) -> str:
    """Generate NaN value report for a given keypoint and individual.

    This function calculates the number and percentage of NaN points
    for a given keypoint and individual in the input data. A keypoint
    is considered NaN if any of its ``space`` coordinates are NaN.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``keypoints`` and ``individuals``
        dimensions.
    keypoint : str
        The name of the keypoint for which to generate the report.
    individual : str, optional
        The name of the individual for which to generate the report.

    Returns
    -------
    str
        A string containing the report.

    """
    selectedta = (
        data.sel(individuals=individual, keypoints=keypoint)
        if individual
        else data.sel(keypoints=keypoint)
    )
    n_nans = selectedta.isnull().any(["space"]).sum(["time"]).item()
    n_points = selectedta.time.size
    percent_nans = round((n_nans / n_points) * 100, 1)
    return f"\n\t\t{keypoint}: {n_nans}/{n_points} ({percent_nans}%)"


def report_nan_values(da: xr.DataArray, label: str | None = None):
    """Report the number and percentage of keypoints that are NaN.

    Numbers are reported for each individual and keypoint in the dataset.

    Parameters
    ----------
    da : xarray.DataArray
        The input data containing pose tracks and metadata.
    label : str, optional
        Label to identify the dataset in the report. If not provided,
        the name of the DataArray is used as the label. Default is None.

    """
    # Compile the report
    if not label:
        label = da.name
    nan_report = f"\nMissing points (marked as NaN) in {label}"
    for ind in da.individuals.values:
        nan_report += f"\n\tIndividual: {ind}"
        for kp in da.keypoints.values:
            nan_report += generate_nan_report(da, kp, individual=ind)
    # Write nan report to logger
    logger = logging.getLogger(__name__)
    logger.info(nan_report)
    # Also print the report to the console
    print(nan_report)


@log_to_attrs
def median_filter(
    data: xr.DataArray,
    window_length: int,
    min_periods: int | None = None,
    print_report: bool = True,
) -> xr.DataArray:
    """Smooth data by applying a median filter over time.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be smoothed.
    window_length : int
        The size of the filter window, representing the fixed number
        of observations used for each window.
    min_periods : int
        Minimum number of observations in the window required to have
        a value (otherwise result is NaN). The default, None, is
        equivalent to setting ``min_periods`` equal to the size of the window.
        This argument is directly  passed to the ``min_periods`` parameter of
        ``xarray.DataArray.rolling``.
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after filtering. Default is ``True``.

    Returns
    -------
    data_smoothed : xarray.DataArray
        The data smoothed using a median filter with the provided parameters.

    Notes
    -----
    By default, whenever one or more NaNs are present in the filter window,
    a NaN is returned to the output array. As a result, any
    stretch of NaNs present in the input data will be propagated
    proportionally to the size of the window  (specifically, by
    ``floor(window_length/2)``). To control this behaviour, the
    ``min_periods`` option can be used to specify the minimum number of
    non-NaN values required in the window to compute a result. For example,
    setting ``min_periods=1`` will result in the filter returning NaNs
    only when all values in the window are NaN, since 1 non-NaN value
    is sufficient to compute the median.

    """
    half_window = window_length // 2
    data_smoothed = (
        data.pad(  # Pad the edges to avoid NaNs
            time=half_window, mode="reflect"
        )
        .rolling(  # Take rolling windows across time
            time=window_length, center=True, min_periods=min_periods
        )
        .median(  # Compute the median of each window
            skipna=True
        )
        .isel(  # Remove the padded edges
            time=slice(half_window, -half_window)
        )
    )

    if print_report:
        report_nan_values(data, "input")
        report_nan_values(data_smoothed, "output")

    return data_smoothed


@log_to_attrs
def savgol_filter(
    data: xr.DataArray,
    window_length: int,
    polyorder: int = 2,
    print_report: bool = True,
    **kwargs,
) -> xr.DataArray:
    """Smooth data by applying a Savitzky-Golay filter over time.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be smoothed.
    window_length : int
        The size of the filter window, representing the fixed number
        of observations used for each window.
    polyorder : int
        The order of the polynomial used to fit the samples. Must be
        less than ``window_length``. By default, a ``polyorder`` of
        2 is used.
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after filtering. Default is ``True``.
    **kwargs : dict
        Additional keyword arguments are passed to scipy.signal.savgol_filter.
        Note that the ``axis`` keyword argument may not be overridden.


    Returns
    -------
    data_smoothed : xarray.DataArray
        The data smoothed using a Savitzky-Golay filter with the
        provided parameters.

    Notes
    -----
    Uses the ``scipy.signal.savgol_filter`` function to apply a Savitzky-Golay
    filter to the input data.
    See the scipy documentation for more information on that function.
    Whenever one or more NaNs are present in a filter window of the
    input data, a NaN is returned to the output array. As a result, any
    stretch of NaNs present in the input data will be propagated
    proportionally to the size of the window (specifically, by
    ``floor(window_length/2)``). Note that, unlike
    ``movement.filtering.median_filter()``, there is no ``min_periods``
    option to control this behaviour.

    """
    if "axis" in kwargs:
        raise log_error(
            ValueError, "The 'axis' argument may not be overridden."
        )
    data_smoothed = data.copy()
    data_smoothed.values = signal.savgol_filter(
        data,
        window_length,
        polyorder,
        axis=0,
        **kwargs,
    )
    if print_report:
        report_nan_values(data, "input")
        report_nan_values(data_smoothed, "output")
    return data_smoothed


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
    data_filtered : xarray.DataArray
        The data where points with a confidence value below the
        user-defined threshold have been converted to NaNs.

    Notes
    -----
    The point-wise confidence values reported by various pose estimation
    frameworks are not standardised, and the range of values can vary.
    For example, DeepLabCut reports a likelihood value between 0 and 1, whereas
    the point confidence reported by SLEAP can range above 1.
    Therefore, the default threshold value will not be appropriate for all
    datasets and does not have the same meaning across pose estimation
    frameworks. We advise users to inspect the confidence values
    in their dataset and adjust the threshold accordingly.

    """
    data_filtered = data.where(confidence >= threshold)
    if print_report:
        report_nan_values(data, "input")
        report_nan_values(data_filtered, "output")
    return data_filtered


@log_to_attrs
def interpolate_over_time(
    data: xr.DataArray,
    method: str = "linear",
    max_gap: int | None = None,
    print_report: bool = True,
) -> xr.DataArray:
    """Fill in NaN values by interpolating over the time dimension.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be interpolated.
    method : str
        String indicating which method to use for interpolation.
        Default is ``linear``. See documentation for
        ``xarray.DataArray.interpolate_na`` for complete list of options.
    max_gap :
        Maximum size of gap, a continuous sequence of NaNs,
        that will be filled. The default value is ``None`` (no limit).
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after interpolation. Default is ``True``.

    Returns
    -------
    data_interpolated : xr.DataArray
        The data where NaN values have been interpolated over
        using the parameters provided.

    """
    data_interpolated = data.interpolate_na(
        dim="time", method=method, max_gap=max_gap, fill_value="extrapolate"
    )
    if print_report:
        report_nan_values(data, "input")
        report_nan_values(data_interpolated, "output")
    return data_interpolated
