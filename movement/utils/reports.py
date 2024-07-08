"""Utility functions for reporting missing data."""

import logging

import xarray as xr

logger = logging.getLogger(__name__)


def calculate_nan_stats(
    data: xr.DataArray, keypoint: str, individual: str | None = None
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
    keypoint : str
        The name of the keypoint for which to generate the report.
    individual : str, optional
        The name of the individual for which to generate the report.

    Returns
    -------
    str
        A string containing the report.

    """
    selected_data = (
        data.sel(individuals=individual, keypoints=keypoint)
        if individual
        else data.sel(keypoints=keypoint)
    )
    n_nans = selected_data.isnull().any(["space"]).sum(["time"]).item()
    n_points = selected_data.time.size
    percent_nans = round((n_nans / n_points) * 100, 1)
    return f"\n\t\t{keypoint}: {n_nans}/{n_points} ({percent_nans}%)"


def report_nan_values(da: xr.DataArray, label: str | None = None):
    """Report the number and percentage of keypoints that are NaN.

    Numbers are reported for each individual and keypoint in the dataset.

    Parameters
    ----------
    da : xarray.DataArray
        The input data containing ``keypoints`` and ``individuals``
        dimensions.
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
            nan_report += calculate_nan_stats(da, kp, individual=ind)
    # Write nan report to logger
    logger.info(nan_report)
    # Also print the report to the console
    print(nan_report)
