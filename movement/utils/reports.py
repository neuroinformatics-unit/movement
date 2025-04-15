"""Utility functions for reporting missing data."""

import numpy as np
import xarray as xr

from movement.utils.logging import logger


def report_nan_values(da: xr.DataArray, label: str | None = None) -> str:
    """Report the number and percentage of keypoints that are NaN.

    Numbers are reported for each individual and keypoint in the data.

    Parameters
    ----------
    da : xarray.DataArray
        The input data containing ``keypoints`` and ``individuals``
        dimensions.
    label : str, optional
        Label to identify the data in the report. If not provided,
        the name of the DataArray is used as the label.
        Default is ``None``.

    Returns
    -------
    str
        A string containing the report.

    """
    label = label or da.name
    nan_report = f"\nMissing points (marked as NaN) in {label}"
    nan_count = (
        da.isnull().any("space").sum("time")
        if "space" in da.dims
        else da.isnull().sum("time")
    )
    # # Drop points without NaNs
    # nan_count = nan_count.where(nan_count > 0, drop=True)
    # if nan_count.size == 0 or nan_count.isnull().all():
    #     return ""
    total_count = da.time.size
    nan_count_str = (
        nan_count.astype(int).astype(str)
        + f"/{total_count} ("
        + (nan_count / total_count * 100).round(2).astype(str)
        + "%)"
    )
    # Stack all dimensions except for the last
    nan_count_df = (
        nan_count_str.stack(new_dim=nan_count_str.dims[:-1])
        if len(nan_count_str.dims) > 1
        else nan_count_str
    ).to_pandas()
    nan_report += f"\n\n{
        (
            nan_count_df.to_string()
            if not isinstance(nan_count_df, np.ndarray)
            else nan_count_df
        )
    }"
    logger.info(nan_report)
    return nan_report
