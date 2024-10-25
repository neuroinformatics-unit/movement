"""Utility functions for reporting missing data."""

import logging

import xarray as xr

logger = logging.getLogger(__name__)


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
    # Check if the data has individuals and keypoints dimensions
    has_individuals_dim = "individuals" in da.dims
    has_keypoints_dim = "keypoints" in da.dims
    # Default values for individuals and keypoints
    individuals = da.individuals.values if has_individuals_dim else [None]
    keypoints = da.keypoints.values if has_keypoints_dim else [None]

    for ind in individuals:
        ind_name = ind if ind is not None else da.individuals.item()
        nan_report += f"\n\tIndividual: {ind_name}"
        for kp in keypoints:
            nan_report += calculate_nan_stats(da, keypoint=kp, individual=ind)
    # Write nan report to logger
    logger.info(nan_report)
    return nan_report
