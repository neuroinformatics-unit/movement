"""Utility functions for reporting missing data."""

import logging

import xarray as xr

logger = logging.getLogger(__name__)


def calculate_nan_stats(
    data: xr.DataArray,
    keypoint: str | None = None,
    individual: str | None = None,
) -> str:
    """Calculate NaN stats for a given keypoint and individual."""
    selection_criteria = {}
    if individual is not None and "individuals" in data.dims:
        selection_criteria["individuals"] = individual
    if keypoint is not None and "keypoints" in data.dims:
        selection_criteria["keypoints"] = keypoint

    selected_data = (
        data.sel(**selection_criteria) if selection_criteria else data
    )

    # Calculate NaNs
    if "space" in selected_data.dims:
        null_mask = selected_data.isnull().any("space")
    else:
        null_mask = selected_data.isnull()

    # Calculate proportion of NaNs
    nan_proportion = null_mask.mean().item()
    percent_nans = round(nan_proportion * 100, 1)

    # Generate label
    label = "data"
    if keypoint:
        label = keypoint
    elif "keypoints" in data.dims:
        label = "all_keypoints"

    return f"\n\t\t{label}: {percent_nans}%"


def report_nan_values(da: xr.DataArray, label: str | None = None) -> str:
    """Report NaN values for all individuals and keypoints."""
    label = label or da.name or "dataset"
    nan_report = f"\nMissing points (marked as NaN) in {label}"
    if "space" in da.dims:
        nan_report += " (any spatial coordinate)"

    # Calculate NaN proportions
    nan_proportion = (
        da.isnull().any("space").mean(dim="time")
        if "space" in da.dims
        else da.isnull().mean(dim="time")
    )

    # Get dimensions to iterate over
    dims = [d for d in nan_proportion.dims if d != "time"]
    if not dims:
        # Handle case with no dimensions except time
        percent = round(nan_proportion.item() * 100, 1)
        nan_report += f"\n\t\tdata: {percent}%"
        logger.info(nan_report)
        return nan_report

    # Handle individuals and keypoints
    if "individuals" in dims:
        for ind in da.individuals.values:
            nan_report += f"\n\tIndividual: {ind}"
            if "keypoints" in dims:
                for kp in da.keypoints.values:
                    percent = round(
                        nan_proportion.sel(
                            individuals=ind, keypoints=kp
                        ).item()
                        * 100,
                        1,
                    )
                    nan_report += f"\n\t\t{kp}: {percent}%"
            else:
                percent = round(
                    nan_proportion.sel(individuals=ind).item() * 100, 1
                )
                nan_report += f"\n\t\tdata: {percent}%"
    else:
        # Only keypoints dimension
        for kp in da.keypoints.values:
            percent = round(nan_proportion.sel(keypoints=kp).item() * 100, 1)
            nan_report += f"\n\t\t{kp}: {percent}%"

    logger.info(nan_report)
    return nan_report
