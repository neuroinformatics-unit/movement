"""Utility functions for reporting missing data."""

import numpy as np
import xarray as xr

from movement.utils.logging import logger
from movement.validators.arrays import validate_dims_coords


def report_nan_values(da: xr.DataArray, label: str | None = None) -> str:
    """Report the number and percentage of data points that are NaN.

    The number of NaNs are counted for each element along the required
    ``time`` dimension.
    If the DataArray has the ``space`` dimension, the ``space`` dimension
    is reduced by checking if any values in the ``space`` coordinates
    are NaN, e.g. a 2D point is considered as NaN if any of its x or y
    coordinates are NaN.

    Parameters
    ----------
    da : xarray.DataArray
        The input data with ``time`` as a required dimension.
    label : str, optional
        Label to identify the data in the report. If not provided,
        the name of the DataArray is used. If the DataArray has no
        name, "data" is used as the label.

    Returns
    -------
    str
        A string containing the report.

    """
    validate_dims_coords(da, {"time": []})
    label = label or da.name or "data"
    nan_report = f"Missing points (marked as NaN) in {label}:"
    nan_count = (
        da.isnull().any("space").sum("time")
        if "space" in da.dims
        else da.isnull().sum("time")
    )
    # Drop coord labels without NaNs
    nan_count = nan_count.where(nan_count > 0, other=0, drop=True)
    if nan_count.size == 0 or nan_count.isnull().all():
        return f"No missing points (marked as NaN) in {label}."
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
    nan_count_df = (
        nan_count_df.to_string()
        if not isinstance(nan_count_df, np.ndarray)
        else nan_count_df
    )
    nan_report += f"\n\n{nan_count_df}"
    logger.info(nan_report)
    return nan_report
