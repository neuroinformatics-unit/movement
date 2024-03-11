import logging
from copy import copy
from datetime import datetime
from functools import wraps
from typing import Union

import numpy as np
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


def filter_diagnostics(ds: xr.Dataset):
    """
    Report the percentage of points that are NaN for each individual and
    each keypoint in the provided dataset.
    """

    # Compile diagnostic report
    diagnostic_report = "\nDatapoints Filtered:\n"
    for ind in ds.individuals.values:
        diagnostic_report += f"\nIndividual: {ind}"
        for kp in ds.keypoints.values:
            n_nans = np.count_nonzero(
                np.isnan(
                    ds.pose_tracks.sel(individuals=ind, keypoints=kp).values[
                        :, 0
                    ]
                )
            )
            n_points = ds.time.values.shape[0]
            prop_nans = round((n_nans / n_points) * 100, 1)
            diagnostic_report += (
                f"\n   {kp}: {n_nans}/{n_points} ({prop_nans}%)"
            )

    # Write diagnostic report to logger
    logger = logging.getLogger(__name__)
    logger.info(diagnostic_report)

    return None


@log_to_attrs
def interpolate_over_time(
    ds: xr.Dataset,
    method: str = "linear",
    max_gap: Union[int, None] = None,
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
        ``xarray.DataSet.interpolate_na`` for complete list of options.
    max_gap :
        The largest gap of consecutive NaNs that will be
        interpolated over. The default value is ``None`` (no limit).

    Returns
    -------
    ds_interpolated : xr.Dataset
        The provided dataset (ds), where NaN values have been
        interpolated over using the parameters provided.
    """
    ds = copy(ds)

    tracks_interpolated = ds.pose_tracks.interpolate_na(
        dim="time", method=method, max_gap=max_gap
    )
    ds_interpolated = ds.update({"pose_tracks": tracks_interpolated})

    return ds_interpolated


@log_to_attrs
def filter_by_confidence(
    ds: xr.Dataset,
    threshold: float = 0.6,
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

    ds = copy(ds)

    tracks_thresholded = ds.pose_tracks.where(ds.confidence >= threshold)
    ds_thresholded = ds.update({"pose_tracks": tracks_thresholded})
    filter_diagnostics(ds_thresholded)

    return ds_thresholded
