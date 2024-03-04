import logging
from copy import copy
from datetime import datetime
from functools import wraps
from typing import Union

import numpy as np
import xarray as xr


def log_to_attrs(func):
    """
    Appends log of the operation performed to xarray.Dataset attributes
    """
    # TODO: Are we okay keeping this decorator here or should this
    #  be refactored to a dedicated `decorators` module?

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
    Reports the number of datapoints filtered
    """
    # TODO: This function currently just counts the number of NaNs in
    #  `pose_tracks` but could potentially be tweaked to deal better
    #  with situations where users use different filters in sequence
    #  and e.g. want to track individual contribution of each filter.

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

    logger = logging.getLogger(__name__)
    logger.info(diagnostic_report)
    print(diagnostic_report)

    return None


@log_to_attrs
def interpolate_over_time(
    ds: xr.Dataset,
    method: str = "linear",
    max_gap: Union[int, None] = None,
) -> Union[xr.Dataset, None]:
    """
    Fills in NaN values by interpolating over the time dimension.

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
        interpolated over. The default value is ``None``.

    Returns
    -------
    ds_thresholded : xr.DataArray
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
    Drops all datapoints where the associated confidence value
    falls below a user-defined threshold.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    threshold : float
        The confidence threshold below which datapoints are filtered.
        A default value of ``0.6`` is used.

    Returns
    -------
    ds_thresholded : xarray.Dataset
        The provided dataset (ds), where datapoints with a confidence
        value below the user-defined threshold have been converted
        to NaNs
    """

    ds = copy(ds)

    tracks_thresholded = ds.pose_tracks.where(ds.confidence >= threshold)
    ds_thresholded = ds.update({"pose_tracks": tracks_thresholded})

    # Diagnostics
    filter_diagnostics(ds_thresholded)

    return ds_thresholded
