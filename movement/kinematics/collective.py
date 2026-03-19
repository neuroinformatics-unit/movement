# collective.py
"""Compute collective behavior metrics for multi-individual tracking data."""

from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr

from movement.utils.logging import logger
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords

_ANGLE_EPS = 1e-12


def compute_polarization(
    data: xr.DataArray,
    body_axis_keypoints: tuple[Hashable, Hashable] | None = None,
    displacement_frames: int = 1,
    return_angle: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    r"""Compute polarization (group alignment) of individuals.

    Polarization measures how aligned the heading directions of individuals
    are. A value of 1 indicates perfect alignment, while a value near 0
    indicates weak or canceling alignment.

    The polarization is computed as

    .. math::

        \Phi = \frac{1}{N} \left\| \sum_{i=1}^{N} \hat{v}_i \right\|

    where :math:`\hat{v}_i` is the unit heading vector for individual
    :math:`i`, and :math:`N` is the number of valid individuals at that time.

    Parameters
    ----------
    data : xarray.DataArray
        Position data. Must contain ``time``, ``space``, and ``individuals`` as
        dimensions. If ``body_axis_keypoints`` is provided, the array must also
        contain a ``keypoints`` dimension. For displacement-based heading,
        pre-select a keypoint (e.g., ``data.sel(keypoints="thorax")``) or the
        first keypoint (index 0) will be used.

        Spatial coordinates must include ``"x"`` and ``"y"``. If additional
        spatial coordinates are present (e.g., ``"z"``), they are ignored;
        polarization is computed in the x/y plane.
    body_axis_keypoints : tuple[Hashable, Hashable], optional
        Pair of keypoint names ``(origin, target)`` used to compute heading as
        the vector from origin to target. If omitted, heading is inferred from
        displacement over ``displacement_frames``.
    displacement_frames : int, default=1
        Number of frames used to compute displacement when
        ``body_axis_keypoints`` is not provided. Must be a positive integer.
        This parameter is ignored when ``body_axis_keypoints`` is provided.
    return_angle : bool, default=False
        If True, also return the mean angle in radians. Returns the mean
        body orientation angle when using ``body_axis_keypoints``, or the
        mean heading angle when using displacement-based polarization.

    Returns
    -------
    xarray.DataArray or tuple[xarray.DataArray, xarray.DataArray]
        If ``return_angle`` is False, returns a DataArray named
        ``"polarization"`` with dimension ``("time",)``.

        If ``return_angle`` is True, returns
        ``(polarization, mean_angle)`` where ``mean_angle`` is a DataArray
        named ``"mean_angle"`` with dimension ``("time",)``.

    Notes
    -----
    Missing data are excluded per individual, per frame.

    Zero-length headings are treated as invalid and excluded from the
    calculation.

    The mean angle is defined from the summed unit-heading vector projected
    onto the x/y plane. When using ``body_axis_keypoints``, this represents
    the mean body orientation; when using displacement, it represents the
    mean movement direction. When no valid headings exist, or when the summed
    heading vector has zero magnitude (for example exact cancellation), the
    returned angle is NaN.

    Examples
    --------
    Compute orientation polarization from body-axis keypoints:

    >>> polarization = compute_polarization(
    ...     ds.position,
    ...     body_axis_keypoints=("tail_base", "neck"),
    ... )

    Compute heading polarization from displacement (pre-select keypoint):

    >>> polarization = compute_polarization(
    ...     ds.position.sel(keypoints="thorax")
    ... )

    If multiple keypoints exist and none is selected, the first is used:

    >>> polarization = compute_polarization(ds.position)

    Return orientation polarization with mean body angle:

    >>> polarization, mean_angle = compute_polarization(
    ...     ds.position,
    ...     body_axis_keypoints=("tail_base", "neck"),
    ...     return_angle=True,
    ... )

    Return heading polarization with mean movement angle:

    >>> polarization, mean_angle = compute_polarization(
    ...     ds.position.sel(keypoints="thorax"),
    ...     return_angle=True,
    ... )

    If multiple keypoints exist, first is used; also return mean angle:

    >>> polarization, mean_angle = compute_polarization(
    ...     ds.position,
    ...     return_angle=True,
    ... )

    """
    _validate_type_data_array(data)
    normalized_keypoints = _validate_position_data(
        data=data,
        body_axis_keypoints=body_axis_keypoints,
    )

    if normalized_keypoints is not None:
        heading_vectors = _compute_heading_from_keypoints(
            data=data,
            body_axis_keypoints=normalized_keypoints,
        )
    else:
        heading_vectors = _compute_heading_from_velocity(
            data=data,
            displacement_frames=displacement_frames,
        )

    heading_xy = _select_xy(heading_vectors)
    norm = compute_norm(heading_xy)
    valid_mask = (~heading_xy.isnull().any(dim="space")) & (norm > 0)

    unit_headings = (heading_xy / norm).where(valid_mask)
    vector_sum = unit_headings.sum(dim="individuals", skipna=True)
    sum_magnitude = compute_norm(vector_sum)
    n_valid = valid_mask.sum(dim="individuals")

    polarization = xr.where(
        n_valid > 0,
        sum_magnitude / n_valid,
        np.nan,
    ).clip(min=0.0, max=1.0)
    polarization = polarization.rename("polarization")

    if not return_angle:
        return polarization

    angle_defined = (n_valid > 0) & (sum_magnitude > _ANGLE_EPS)
    mean_angle = xr.where(
        angle_defined,
        np.arctan2(
            vector_sum.sel(space="y"),
            vector_sum.sel(space="x"),
        ),
        np.nan,
    ).rename("mean_angle")

    return polarization, mean_angle


def _compute_heading_from_keypoints(
    data: xr.DataArray,
    body_axis_keypoints: tuple[Hashable, Hashable],
) -> xr.DataArray:
    """Compute heading vectors from two keypoints (origin to target)."""
    origin, target = body_axis_keypoints
    heading = data.sel(keypoints=target, drop=True) - data.sel(
        keypoints=origin,
        drop=True,
    )
    return heading


def _compute_heading_from_velocity(
    data: xr.DataArray,
    displacement_frames: int = 1,
) -> xr.DataArray:
    """Compute heading vectors from displacement direction."""
    _validate_displacement_frames(displacement_frames)

    position = data
    if "keypoints" in data.dims:
        if data.sizes["keypoints"] < 1:
            raise ValueError(
                "data.keypoints must contain at least one keypoint."
            )
        position = data.isel(keypoints=0, drop=True)

        if "keypoints" in data.coords and data.coords["keypoints"].size > 0:
            logger.info(
                "Using keypoint '%s' for displacement-based heading.",
                data.coords["keypoints"].values[0],
            )
        else:
            logger.info(
                "Using keypoint index 0 for displacement-based heading."
            )

    displacement = position - position.shift(time=displacement_frames)
    return displacement


def _select_xy(data: xr.DataArray) -> xr.DataArray:
    """Select the planar x/y components and return standard dim order."""
    return data.sel(space=["x", "y"]).transpose("time", "space", "individuals")


def _validate_position_data(
    data: xr.DataArray,
    body_axis_keypoints: tuple[Hashable, Hashable] | None,
) -> tuple[Hashable, Hashable] | None:
    """Validate the input array and normalize ``body_axis_keypoints``."""
    validate_dims_coords(
        data,
        {
            "time": [],
            "space": [],
            "individuals": [],
        },
    )

    allowed_dims = {"time", "space", "individuals", "keypoints"}
    unexpected_dims = set(data.dims) - allowed_dims
    if unexpected_dims:
        raise ValueError(
            f"data contains unsupported dimension(s): "
            f"{sorted(str(d) for d in unexpected_dims)}"
        )

    if "space" not in data.coords:
        raise ValueError(
            "data must have coordinate labels for the 'space' dimension."
        )

    space_labels = set(data.coords["space"].values.tolist())
    if not {"x", "y"}.issubset(space_labels):
        raise ValueError(
            "data.space must include coordinate labels 'x' and 'y'."
        )

    if body_axis_keypoints is None:
        return None

    origin, target = _normalize_body_axis_keypoints(body_axis_keypoints)

    if "keypoints" not in data.dims:
        raise ValueError(
            "body_axis_keypoints requires a 'keypoints' dimension in data."
        )

    validate_dims_coords(data, {"keypoints": [origin, target]})
    return origin, target


def _normalize_body_axis_keypoints(
    body_axis_keypoints: tuple[Hashable, Hashable] | Any,
) -> tuple[Hashable, Hashable]:
    """Validate and normalize the keypoint pair."""
    if isinstance(body_axis_keypoints, (str, bytes)):
        raise TypeError(
            "body_axis_keypoints must be an iterable of exactly two "
            "keypoint names."
        )

    try:
        origin, target = body_axis_keypoints
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "body_axis_keypoints must be an iterable of exactly two "
            "keypoint names."
        ) from exc

    for keypoint in (origin, target):
        if not isinstance(keypoint, Hashable):
            raise TypeError("Each body axis keypoint must be hashable.")

    if origin == target:
        raise ValueError(
            "body_axis_keypoints must contain two distinct keypoint names."
        )

    return origin, target


def _validate_displacement_frames(displacement_frames: int) -> None:
    """Validate the displacement window."""
    if isinstance(displacement_frames, (bool, np.bool_)) or not isinstance(
        displacement_frames,
        (int, np.integer),
    ):
        raise TypeError("displacement_frames must be a positive integer.")

    if displacement_frames < 1:
        raise ValueError("displacement_frames must be >= 1.")


def _validate_type_data_array(data: xr.DataArray) -> None:
    """Validate that the input is an xarray.DataArray."""
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Input data must be an xarray.DataArray, but got {type(data)}."
        )
