"""Compute collective behavior metrics for multi-individual tracking data."""

from collections.abc import Hashable

import numpy as np
import xarray as xr

from movement.kinematics.kinematics import compute_velocity
from movement.utils.logging import logger
from movement.utils.vector import compute_norm, convert_to_unit
from movement.validators.arrays import validate_dims_coords


def compute_polarization(
    data: xr.DataArray,
    heading_keypoints: tuple[Hashable, Hashable] | None = None,
) -> xr.DataArray:
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
        If None, heading is inferred from the velocity of the first
        available keypoint.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the polarization value at each
        time point, with dimensions ``(time,)``. Values range from 0
        (random orientations) to 1 (perfectly aligned).

    Notes
    -----
    If ``heading_keypoints`` is provided, the heading for each individual
    is computed as the unit vector from the origin to the target
    keypoint. If not provided, heading is inferred from the instantaneous
    velocity direction.

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

    Compute polarization using velocity-inferred heading:

    >>> polarization = compute_polarization(ds.position)

    See Also
    --------
    movement.kinematics.compute_velocity : Compute velocity from position.

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
        heading_vectors = _compute_heading_from_velocity(data)

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


def _compute_heading_from_velocity(data: xr.DataArray) -> xr.DataArray:
    """Compute heading vectors from velocity (displacement direction).

    Uses the first available keypoint if multiple are present.

    Parameters
    ----------
    data : xarray.DataArray
        Position data with ``time`` dimension.

    Returns
    -------
    xarray.DataArray
        Heading vectors based on velocity direction.

    """
    # If keypoints dimension exists, use first keypoint
    if "keypoints" in data.dims:
        first_keypoint = data.keypoints.values[0]
        position = data.sel(keypoints=first_keypoint, drop=True)
        logger.info(
            f"Using keypoint '{first_keypoint}' for velocity-based heading."
        )
    else:
        position = data

    # Compute velocity as heading direction
    velocity = compute_velocity(position)

    return velocity


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
