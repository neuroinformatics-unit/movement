"""Compute collective behavior metrics for multi-individual tracking data."""

from collections.abc import Hashable

import numpy as np
import xarray as xr

from movement.utils.logging import logger
from movement.utils.vector import compute_norm, convert_to_unit
from movement.validators.arrays import validate_dims_coords


def compute_polarization(
    data: xr.DataArray,
    heading_keypoints: tuple[Hashable, Hashable] | None = None,
    displacement_frames: int = 1,
    return_angle: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    r"""Compute the polarization (group alignment) of multiple individuals.

    Polarization measures how aligned the heading directions of individuals
    are. A value of 1 indicates all individuals are heading in the same
    direction, while a value near 0 indicates random orientations.

    The polarization is computed as:

    .. math::  \Phi = \frac{1}{N} \left\| \sum_{i=1}^{N} \hat{v}_i \right\|

    where :math:`\hat{v}_i` is the unit heading vector for individual
    :math:`i`, and :math:`N` is the number of individuals.

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. Must contain ``time``,
        ``space``, and ``individuals`` as dimensions. The ``keypoints``
        dimension is required only if ``heading_keypoints`` is provided.
    heading_keypoints : tuple of Hashable, optional
        A tuple of two keypoint names ``(origin, target)`` used to
        compute the heading direction as the vector from origin to
        target (e.g., ``("neck", "nose")`` or ``("tail", "head")``).
        If None, heading is inferred from the displacement of the first
        available keypoint.
    displacement_frames : int, optional
        Number of frames over which to compute displacement when
        ``heading_keypoints`` is None. Default is 1 (frame-to-frame
        displacement). Use higher values for smoother heading estimates
        (e.g., fps value for 1-second displacement). This parameter is
        ignored when ``heading_keypoints`` is provided.
    return_angle : bool, optional
        If True, also return the mean heading angle (in radians) of the
        group at each time point. Default is False.

    Returns
    -------
    xarray.DataArray or tuple of xarray.DataArray
        If ``return_angle`` is False (default), returns an xarray DataArray
        containing the polarization value at each time point, with
        dimensions ``(time,)``. Values range from 0 (random orientations)
        to 1 (perfectly aligned).

        If ``return_angle`` is True, returns a tuple of two DataArrays:
        ``(polarization, mean_angle)``, where ``mean_angle`` contains the
        mean heading direction in radians at each time point.

    Notes
    -----
    If ``heading_keypoints`` is provided, the heading for each individual
    is computed as the unit vector from the origin to the target
    keypoint. If not provided, heading is inferred from the displacement
    direction over ``displacement_frames`` frames.

    Frames where an individual has missing data (NaN) are handled by
    excluding that individual from the polarization calculation for that
    frame.

    Examples
    --------
    Compute polarization using two keypoints to define heading:

    >>> polarization = compute_polarization(
    ...     ds.position,
    ...     heading_keypoints=("neck", "nose"),
    ... )

    Compute polarization using displacement-inferred heading:

    >>> polarization = compute_polarization(ds.position)

    Compute polarization with 1-second displacement (at 30 fps):

    >>> polarization = compute_polarization(
    ...     ds.position,
    ...     displacement_frames=30,
    ... )

    Also return the mean heading angle:

    >>> polarization, mean_angle = compute_polarization(
    ...     ds.position,
    ...     return_angle=True,
    ... )

    """
    # Validate input data
    _validate_type_data_array(data)
    validate_dims_coords(
        data,
        {
            "time": [],
            "space": [],
            "individuals": [],
        },
    )

    # Compute heading vectors for all individuals
    if heading_keypoints is not None:
        heading_vectors = _compute_heading_from_keypoints(
            data, heading_keypoints
        )
    else:
        heading_vectors = _compute_heading_from_velocity(
            data, displacement_frames=displacement_frames
        )

    # Convert to unit vectors
    unit_headings = convert_to_unit(heading_vectors)

    # Sum unit vectors across individuals
    # Use nansum to handle missing data
    vector_sum = unit_headings.sum(dim="individuals", skipna=True)

    # Count valid (non-NaN) individuals per time point
    # A heading is valid if both x and y are not NaN
    valid_mask = ~unit_headings.isnull().any(dim="space")
    n_valid = valid_mask.sum(dim="individuals")

    # Compute magnitude of the sum
    sum_magnitude = compute_norm(vector_sum)

    # Normalize by number of valid individuals
    # Avoid division by zero
    polarization = xr.where(n_valid > 0, sum_magnitude / n_valid, np.nan)

    polarization.name = "polarization"

    if return_angle:
        # Compute mean heading angle from the vector sum
        # arctan2(y, x) gives angle in radians
        mean_angle = np.arctan2(
            vector_sum.sel(space="y"),
            vector_sum.sel(space="x"),
        )
        mean_angle = xr.where(n_valid > 0, mean_angle, np.nan)
        mean_angle.name = "mean_angle"
        return polarization, mean_angle

    return polarization


def _compute_heading_from_keypoints(
    data: xr.DataArray,
    heading_keypoints: tuple[Hashable, Hashable],
) -> xr.DataArray:
    """Compute heading vectors from two keypoints (origin to target).

    Parameters
    ----------
    data : xarray.DataArray
        Position data with ``keypoints`` dimension.
    heading_keypoints : tuple of Hashable
        A tuple of ``(origin, target)`` keypoint names. The heading
        vector points from origin toward target.

    Returns
    -------
    xarray.DataArray
        Heading vectors with dimensions ``(time, space, individuals)``.

    """
    origin, target = heading_keypoints

    # Validate keypoints are different
    if origin == target:
        raise logger.error(
            ValueError("The origin and target keypoints may not be identical.")
        )

    # Validate keypoints exist
    validate_dims_coords(
        data,
        {"keypoints": [origin, target]},
    )

    # Compute heading as vector from origin to target
    heading = data.sel(keypoints=target, drop=True) - data.sel(
        keypoints=origin, drop=True
    )

    return heading


def _compute_heading_from_velocity(
    data: xr.DataArray,
    displacement_frames: int = 1,
) -> xr.DataArray:
    """Compute heading vectors from displacement direction.

    Uses the first available keypoint if multiple are present.

    Parameters
    ----------
    data : xarray.DataArray
        Position data with ``time`` dimension.
    displacement_frames : int, optional
        Number of frames over which to compute displacement. Default is 1
        (frame-to-frame displacement). Use higher values for smoother
        heading estimates (e.g., fps for 1-second displacement).

    Returns
    -------
    xarray.DataArray
        Heading vectors based on displacement direction.

    """
    # If keypoints dimension exists, use first keypoint
    if "keypoints" in data.dims:
        first_keypoint = data.keypoints.values[0]
        position = data.sel(keypoints=first_keypoint, drop=True)
        logger.info(
            f"Using keypoint '{first_keypoint}' for displacement-based heading."
        )
    else:
        position = data

    # Compute displacement over N frames
    # displacement[t] = position[t] - position[t - displacement_frames]
    displacement = position - position.shift(time=displacement_frames)

    return displacement


def _validate_type_data_array(data: xr.DataArray) -> None:
    """Validate the input data is an xarray DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to validate.

    Raises
    ------
    TypeError
        If the input data is not an xarray DataArray.

    """
    if not isinstance(data, xr.DataArray):
        raise logger.error(
            TypeError(
                "Input data must be an xarray.DataArray, "
                f"but got {type(data)}."
            )
        )
