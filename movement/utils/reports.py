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
    # Only select dimensions that exist in the DataArray

    # Handle existing dimensions
    if individual is not None and "individuals" in data.dims:
        selection_criteria["individuals"] = individual
    if keypoint is not None and "keypoints" in data.dims:
        selection_criteria["keypoints"] = keypoint

    selected_data = (
        data.sel(**selection_criteria) if selection_criteria else data
    )

    # Calculate NaNs with dimension-agnostic approach
    if "space" in selected_data.dims:
        null_mask = selected_data.isnull().any("space")
    else:
        null_mask = selected_data.isnull()

    # Sum all dimensions except time to get scalar
    n_nans = null_mask.sum().item()
    n_points = selected_data.time.size
    percent_nans = (
        round((n_nans / n_points) * 100, 1) if n_points != 0 else 0.0
    )

    # Generate label based on keypoint presence
    # For single keypoint, don't include the name in the report
    if "keypoints" in data.dims and keypoint and len(data.keypoints) > 1:
        label = keypoint
    else:
        label = "data"

    return f"\n\t\t{label}: {n_nans}/{n_points} ({percent_nans}%)"


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
    label = label or da.name or "dataset"
    nan_report = f"\nMissing points (marked as NaN) in {label}"

    # Add space dimension note if present
    if "space" in da.dims:
        nan_report += " (any spatial coordinate)"

    # Use coords to get only individuals/keypoints present in the DataArray
    # This ensures correct handling of subsets
    if "individuals" in da.dims:
        individuals = da.coords["individuals"].values
    else:
        individuals = [None]

    if "keypoints" in da.dims:
        keypoints = da.coords["keypoints"].values
        # Only explicitly list keypoints if more than one exists
        if len(keypoints) <= 1:
            keypoints = [None]
    else:
        keypoints = [None]

    for ind in individuals:
        if "individuals" in da.dims:
            nan_report += f"\n\tIndividual: {ind}"

        for kp in keypoints:
            use_kp = kp if "keypoints" in da.dims and len(da.keypoints) > 1 else None
            nan_report += calculate_nan_stats(
                da,
                keypoint=use_kp,
                individual=ind if "individuals" in da.dims else None,
            )

    logger.info(nan_report)
    return nan_report
