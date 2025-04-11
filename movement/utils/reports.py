"""Utility functions for reporting missing data."""

import xarray as xr

from movement.utils.logging import logger


def calculate_nan_stats(
    data: xr.DataArray,
    keypoint: str | None = None,
    individual: str | None = None,
) -> str:
    """Calculate NaN stats for a given keypoint and individual.

    This function calculates the number and percentage of NaN points
    for a given keypoint and individual in the input data. A keypoint
    is considered NaN if any of its ``space`` coordinates are NaN.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``keypoints`` and ``individuals``
        dimensions.
    keypoint : str, optional
        The name of the keypoint for which to generate the report.
        If ``None``, it is assumed that the input data contains only
        one keypoint and this keypoint is used.
        Default is ``None``.
    individual : str, optional
        The name of the individual for which to generate the report.
        If ``None``, it is assumed that the input data contains only
        one individual and this individual is used.
        Default is ``None``.

    Returns
    -------
    str
        A string containing the report.

    """
    selection_criteria = {}
    if individual is not None:
        selection_criteria["individuals"] = individual
    if keypoint is not None:
        selection_criteria["keypoints"] = keypoint
    selected_data = (
        data.sel(**selection_criteria) if selection_criteria else data
    )
    n_nans = selected_data.isnull().any(["space"]).sum(["time"]).item()
    n_points = selected_data.time.size
    percent_nans = round((n_nans / n_points) * 100, 1)
    return f"\n\t\t{keypoint}: {n_nans}/{n_points} ({percent_nans}%)"


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
    # Compile the report
    label = label or da.name
    nan_report = f"\nMissing points (marked as NaN) in {label}"
    nan_count = (
        da.isnull().any(["space"]).sum(dim="time")
        if "space" in da.dims
        else da.isnull().sum(dim="time")
    )
    # Drop points without NaNs
    nan_count = nan_count.where(nan_count > 0, drop=True)
    if nan_count.size == 0:
        return ""
    total_count = da.time.size
    nan_stats_str = (
        nan_count.astype(int).astype(str)
        + f"/{total_count} ("
        + (nan_count / total_count * 100).round(2).astype(str)
        + "%)"
    )
    # Stack all dimensions except for the last
    nan_stats_df = nan_stats_str.stack(
        new_dim=nan_stats_str.dims[:-1]
    ).to_pandas()
    nan_report += f"\n\n{nan_stats_df.to_string()}"
    # Write nan report to logger
    logger.info(nan_report)
    return nan_report
