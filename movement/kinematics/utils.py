# D:\Programs\NIU\movement\movement\kinematics\utils.py
"""Utility functions for kinematics computations."""

import xarray as xr

from movement.kinematics.motion import compute_displacement
from movement.utils.logging import log_error, log_warning
from movement.utils.reports import report_nan_values
from movement.utils.vector import compute_norm


def _validate_type_data_array(data: xr.DataArray) -> None:
    """Validate the input data is an xarray DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to validate.

    Raises
    ------
    ValueError
        If the input data is not an xarray DataArray.

    """
    if not isinstance(data, xr.DataArray):
        raise log_error(
            TypeError,
            f"Input data must be an xarray.DataArray, but got {type(data)}.",
        )


def _compute_scaled_path_length(data: xr.DataArray) -> xr.DataArray:
    """Compute scaled path length based on proportion of valid segments.

    Path length is first computed based on valid segments (non-NaN values
    on both ends of the segment) and then scaled based on the proportion of
    valid segments per point track - i.e. the result is divided by the
    proportion of valid segments.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed path length,
        with dimensions matching those of the input data,
        except ``time`` and ``space`` are removed.

    """
    displacement = compute_displacement(data).isel(time=slice(1, None))
    valid_segments = (~displacement.isnull()).all(dim="space").sum(dim="time")
    valid_proportion = valid_segments / (data.sizes["time"] - 1)
    return compute_norm(displacement).sum(dim="time") / valid_proportion


def _warn_about_nan_proportion(
    data: xr.DataArray, nan_warn_threshold: float
) -> None:
    """Print a warning if the proportion of NaN values exceeds a threshold.

    The NaN proportion is evaluated per point track, and a given point is
    considered NaN if any of its ``space`` coordinates are NaN. The warning
    specifically lists the point tracks that exceed the threshold.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array.
    nan_warn_threshold : float
        The threshold for the proportion of NaN values. Must be a number
        between 0 and 1.

    """
    nan_warn_threshold = float(nan_warn_threshold)
    if not 0 <= nan_warn_threshold <= 1:
        raise log_error(
            ValueError,
            "nan_warn_threshold must be between 0 and 1.",
        )
    n_nans = data.isnull().any(dim="space").sum(dim="time")
    data_to_warn_about = data.where(
        n_nans > data.sizes["time"] * nan_warn_threshold, drop=True
    )
    if len(data_to_warn_about) > 0:
        log_warning(
            "The result may be unreliable for point tracks with many "
            "missing values. The following tracks have more than "
            f"{nan_warn_threshold * 100:.3} % NaN values:"
        )
        print(report_nan_values(data_to_warn_about))


def _validate_labels_dimension(data: xr.DataArray, dim: str) -> xr.DataArray:
    """Validate the input data contains the ``dim`` for labelling dimensions.

    This function ensures the input data contains the ``dim``
    used as labels (coordinates) when applying
    :func:`scipy.spatial.distance.cdist` to
    the input data, by adding a temporary dimension if necessary.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to validate.
    dim : str
        The dimension to validate.

    Returns
    -------
    xarray.DataArray
        The input data with the labels dimension validated.

    """
    if data.coords.get(dim) is None:
        data = data.assign_coords({dim: "temp_dim"})
    if data.coords[dim].ndim == 0:
        data = data.expand_dims(dim).transpose("time", "space", dim)
    return data
