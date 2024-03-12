import logging
from datetime import datetime
from functools import wraps
from typing import Union

import xarray as xr


def log_to_attrs(func):
    """
    Decorator that logs the operation performed by the wrapped function
    and appends the log entry to the xarray.Dataset's "log" attribute.
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
            if "log" not in result.attrs.keys():
                result.attrs["log"] = []
            result.attrs["log"].append(log_entry)

        return result

    return wrapper


def report_nan_values(ds: xr.Dataset, ds_label: str = "dataset"):
    """
    Report the number and percentage of points that are NaN for each individual
    and each keypoint in the provided dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    ds_label : str
        Label to identify the dataset in the report. Default is "dataset".
    """

    # Compile the report
    nan_report = f"\nMissing points (marked as NaN) in {ds_label}:"
    for ind in ds.individuals.values:
        nan_report += f"\n\tIndividual: {ind}"
        for kp in ds.keypoints.values:
            # Get the track for the current individual and keypoint
            track_ = ds.pose_tracks.sel(individuals=ind, keypoints=kp)
            # A point is considered NaN if any of its space coordinates are NaN
            n_nans = track_.isnull().any(["space"]).sum(["time"]).item()
            n_points = track_.time.size
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
    """
    Fill in NaN values by interpolating over the time dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
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
    poses_interpolated = ds.pose_tracks.interpolate_na(
        dim="time", method=method, max_gap=max_gap, fill_value="extrapolate"
    )
    ds_interpolated.update({"pose_tracks": poses_interpolated})
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
    """
    Drop all points where the associated confidence value falls below a
    user-defined threshold.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
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
        to NaNs

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
        {"pose_tracks": ds.pose_tracks.where(ds.confidence >= threshold)}
    )
    if print_report:
        report_nan_values(ds, "input dataset")
        report_nan_values(ds_thresholded, "filtered dataset")

    return ds_thresholded
