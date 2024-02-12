from datetime import datetime
from typing import Union

import numpy as np
import xarray as xr


def interp_pose(
    ds: xr.Dataset,
    method: str = "linear",
    limit: Union[int, None] = None,
    max_gap: Union[int, None] = None,
    inplace: bool = False,
) -> Union[xr.Dataset, None]:
    """
    Fills in NaN values by interpolating over the time dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    method : str
        String indicating which method to use for interpolation.
        Default is `linear`. See documentation for
        `xarray.DataSet.interpolate_na` for complete list of options.
    limit : int | None
        Maximum number of consecutive NaNs to interpolate over.
        `None` indicates no limit, and is the default value.
    max_gap : TODO: Clarify the difference between `limit` & `max_gap`
        The largest gap of consecutive NaNs that will be
        interpolated over. The default value is `None`.
    inplace: bool
        If true, updates the provided DataSet in place and returns
        `None`.

    Returns
    -------
    ds_thresholded : xr.DataArray
        The provided dataset (ds), where NaN values have been
        interpolated over using the parameters provided.
    """
    # TODO: This method interpolates over confidence values as well.
    #  -> Figure out whether this is the desired default behavior.
    ds_interpolated = ds.interpolate_na(
        dim="time", method=method, limit=limit, max_gap=max_gap
    )

    # Logging
    log_entry = {
        "operation": "interp_pose",
        "method": method,
        "limit": limit,
        "max_gap": max_gap,
        "inplace": inplace,
        "datetime": str(datetime.now()),
    }
    ds_interpolated.attrs["log"].append(log_entry)

    if inplace:
        ds["pose_tracks"] = ds_interpolated["pose_tracks"]
        ds["confidence"] = ds_interpolated["confidence"]
        return None
    else:
        return ds_interpolated


def filter_confidence(
    ds: xr.Dataset,
    threshold: float = 0.6,
    inplace: bool = False,
    interp: bool = False,
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
        A default value of `0.6` is used.
    inplace : bool
        If true, updates the provided DataSet in place and returns
        `None`.
    interp : bool
        If true, NaNs are interpolated over using `interp_pose` with
        default parameters.

    Returns
    -------
    ds_thresholded : xarray.Dataset
        The provided dataset (ds), where datapoints with a confidence
        value below the user-defined threshold have been converted
        to NaNs
    """

    ds_thresholded = ds.where(ds.confidence >= threshold)

    # Diagnostics
    print("\nDatapoints Filtered:\n")
    for kp in ds.keypoints.values:
        n_nans = np.count_nonzero(
            np.isnan(ds_thresholded.confidence.sel(keypoints=f"{kp}").values)
        )
        n_points = ds.time.values.shape[0]
        prop_nans = round((n_nans / n_points) * 100, 2)
        print(f"{kp}: {n_nans}/{n_points} ({prop_nans}%)")

    # TODO: Is this enough diagnostics? Should I write logic to allow
    #  users to optionally plot out confidence distributions + imposed
    #  threshold?

    # Logging
    if "log" not in ds_thresholded.attrs.keys():
        ds_thresholded.attrs["log"] = []

    log_entry = {
        "operation": "filter_confidence",
        "threshold": threshold,
        "inplace": inplace,
        "datetime": str(datetime.now()),
    }
    ds_thresholded.attrs["log"].append(log_entry)

    # Interpolation
    if interp:
        interp_pose(ds_thresholded, inplace=True)

    if inplace:
        ds["pose_tracks"] = ds_thresholded["pose_tracks"]
        ds["confidence"] = ds_thresholded["confidence"]
        return None
    if not inplace:
        return ds_thresholded