"""Functions for filtering and interpolating pose tracks in xarray datasets."""

import logging
from datetime import datetime
from functools import wraps
from typing import Optional, Union

import xarray as xr
from scipy import signal

from movement.logging import log_error


def log_to_attrs(func):
    """Log the operation performed by the wrapped function.

    This decorator appends log entries to the xarray.Dataset's "log" attribute.
    For the decorator to work, the wrapped function must accept an
    xarray.Dataset as its first argument and return an xarray.Dataset.
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


def report_nan_values(ds: xr.Dataset, ds_label: str = "dataset"):
    """Report the number and percentage of points that are NaN.

    Numbers are reported for each individual and keypoint in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing position, confidence scores, and metadata.
    ds_label : str
        Label to identify the dataset in the report. Default is "dataset".

    """
    # Compile the report
    nan_report = f"\nMissing points (marked as NaN) in {ds_label}:"
    for ind in ds.individuals.values:
        nan_report += f"\n\tIndividual: {ind}"
        for kp in ds.keypoints.values:
            # Get the position track for the current individual and keypoint
            position = ds.position.sel(individuals=ind, keypoints=kp)
            # A point is considered NaN if any of its space coordinates are NaN
            n_nans = position.isnull().any(["space"]).sum(["time"]).item()
            n_points = position.time.size
            percent_nans = round((n_nans / n_points) * 100, 1)
            nan_report += f"\n\t\t{kp}: {n_nans}/{n_points} ({percent_nans}%)"

    # Write nan report to logger
    logger = logging.getLogger(__name__)
    logger.info(nan_report)
    # Also print the report to the console
    print(nan_report)
    return None


@log_to_attrs
def interpolate_over_time(
    ds: xr.Dataset,
    method: str = "linear",
    max_gap: Union[int, None] = None,
    print_report: bool = True,
) -> Union[xr.Dataset, None]:
    """Fill in NaN values by interpolating over the time dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing position, confidence scores, and metadata.
    method : str
        String indicating which method to use for interpolation.
        Default is ``linear``. See documentation for
        ``xarray.DataArray.interpolate_na`` for complete list of options.
    max_gap :
        The largest time gap of consecutive NaNs (in seconds) that will be
        interpolated over. The default value is ``None`` (no limit).
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after interpolation. Default is ``True``.

    Returns
    -------
    ds_interpolated : xr.Dataset
        The provided dataset (ds), where NaN values have been
        interpolated over using the parameters provided.

    """
    ds_interpolated = ds.copy()
    position_interpolated = ds.position.interpolate_na(
        dim="time", method=method, max_gap=max_gap, fill_value="extrapolate"
    )
    ds_interpolated.update({"position": position_interpolated})
    if print_report:
        report_nan_values(ds, "input dataset")
        report_nan_values(ds_interpolated, "interpolated dataset")
    return ds_interpolated


@log_to_attrs
def filter_by_confidence(
    ds: xr.Dataset,
    threshold: float = 0.6,
    print_report: bool = True,
) -> Union[xr.Dataset, None]:
    """Drop all points below a certain confidence threshold.

    Position points with an associated confidence value below the threshold are
    converted to NaN.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing position, confidence scores, and metadata.
    threshold : float
        The confidence threshold below which datapoints are filtered.
        A default value of ``0.6`` is used. See notes for more information.
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after filtering. Default is ``True``.

    Returns
    -------
    ds_thresholded : xarray.Dataset
        The provided dataset (ds), where points with a confidence
        value below the user-defined threshold have been converted
        to NaNs.

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
    ds_thresholded = ds.copy()
    ds_thresholded.update(
        {"position": ds.position.where(ds.confidence >= threshold)}
    )
    if print_report:
        report_nan_values(ds, "input dataset")
        report_nan_values(ds_thresholded, "filtered dataset")

    return ds_thresholded


@log_to_attrs
def median_filter(
    ds: xr.Dataset,
    window_length: int,
    min_periods: Optional[int] = None,
    print_report: bool = True,
) -> xr.Dataset:
    """Smooth pose tracks by applying a median filter over time.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing position, confidence scores, and metadata.
    window_length : int
        The size of the filter window. Window length is interpreted
        as being in the input dataset's time unit, which can be inspected
        with ``ds.time_unit``.
    min_periods : int
        Minimum number of observations in window required to have a value
        (otherwise result is NaN). The default, None, is equivalent to
        setting ``min_periods`` equal to the size of the window.
        This argument is directly  passed to the ``min_periods`` parameter of
        ``xarray.DataArray.rolling``.
    print_report : bool
        Whether to print a report on the number of NaNs in the dataset
        before and after filtering. Default is ``True``.

    Returns
    -------
    ds_smoothed : xarray.Dataset
        The provided dataset (ds), where pose tracks have been smoothed
        using a median filter with the provided parameters.

    Notes
    -----
    By default, whenever one or more NaNs are present in the filter window,
    a NaN is returned to the output array. As a result, any
    stretch of NaNs present in the input dataset will be propagated
    proportionally to the size of the window in frames  (specifically, by
    ``floor(window_length/2)``). To control this behaviour, the
    ``min_periods`` option can be used to specify the minimum number of
    non-NaN values required in the window to compute a result. For example,
    setting ``min_periods=1`` will result in the filter returning NaNs
    only when all values in the window are NaN, since 1 non-NaN value
    is sufficient to compute the median.

    """
    ds_smoothed = ds.copy()

    # Express window length (and its half) in frames
    if ds.time_unit == "seconds":
        window_length = int(window_length * ds.fps)

    half_window = window_length // 2

    ds_smoothed.update(
        {
            "position": ds.position.pad(  # Pad the edges to avoid NaNs
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
        }
    )

    if print_report:
        report_nan_values(ds, "input dataset")
        report_nan_values(ds_smoothed, "filtered dataset")

    return ds_smoothed


@log_to_attrs
def savgol_filter(
    ds: xr.Dataset,
    window_length: int,
    polyorder: int = 2,
    print_report: bool = True,
    **kwargs,
) -> xr.Dataset:
    """Smooth pose tracks by applying a Savitzky-Golay filter over time.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing position, confidence scores, and metadata.
    window_length : int
        The size of the filter window. Window length is interpreted
        as being in the input dataset's time unit, which can be inspected
        with ``ds.time_unit``.
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
    ds_smoothed : xarray.Dataset
        The provided dataset (ds), where pose tracks have been smoothed
        using a Savitzky-Golay filter with the provided parameters.

    Notes
    -----
    Uses the ``scipy.signal.savgol_filter`` function to apply a Savitzky-Golay
    filter to the input dataset's ``position`` variable.
    See the scipy documentation for more information on that function.
    Whenever one or more NaNs are present in a filter window of the
    input dataset, a NaN is returned to the output array. As a result, any
    stretch of NaNs present in the input dataset will be propagated
    proportionally to the size of the window in frames (specifically, by
    ``floor(window_length/2)``). Note that, unlike
    ``movement.filtering.median_filter()``, there is no ``min_periods``
    option to control this behaviour.

    """
    if "axis" in kwargs:
        raise log_error(
            ValueError, "The 'axis' argument may not be overridden."
        )

    ds_smoothed = ds.copy()

    if ds.time_unit == "seconds":
        window_length = int(window_length * ds.fps)

    position_smoothed = signal.savgol_filter(
        ds.position,
        window_length,
        polyorder,
        axis=0,
        **kwargs,
    )
    position_smoothed_da = ds.position.copy(data=position_smoothed)

    ds_smoothed.update({"position": position_smoothed_da})

    if print_report:
        report_nan_values(ds, "input dataset")
        report_nan_values(ds_smoothed, "filtered dataset")

    return ds_smoothed
