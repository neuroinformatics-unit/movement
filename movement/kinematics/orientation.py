"""Compute orientations as vectors and angles."""

from collections.abc import Hashable
from typing import Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.kinematics.kinematics import compute_time_derivative
from movement.utils.logging import logger
from movement.utils.vector import (
    compute_norm,
    compute_signed_angle_2d,
    convert_to_unit,
)
from movement.validators.arrays import validate_dims_coords


def compute_forward_vector(
    data: xr.DataArray,
    left_keypoint: Hashable,
    right_keypoint: Hashable,
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
) -> xr.DataArray:
    """Compute a 2D forward vector given two left-right symmetric keypoints.

    The forward vector is computed as a vector perpendicular to the
    line connecting two symmetrical keypoints on either side of the body
    (i.e., symmetrical relative to the mid-sagittal plane), and pointing
    forwards (in the rostral direction). A top-down or bottom-up view of the
    animal is assumed (see Notes).

    Parameters
    ----------
    data
        The input data representing position. This must contain
        the two symmetrical keypoints located on the left and
        right sides of the body, respectively.
    left_keypoint
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint
        Name of the right keypoint, e.g., "right_ear"
    camera_view
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the forward vector, with
        dimensions matching the input data array, but without the
        ``keypoint`` dimension.

    Notes
    -----
    To determine the forward direction of the animal, we need to specify
    (1) the right-to-left direction of the animal and (2) its upward direction.
    We determine the right-to-left direction via the input left and right
    keypoints. The upwards direction, in turn, can be determined by passing the
    ``camera_view`` argument with either ``"top_down"`` or ``"bottom_up"``. If
    the camera view is specified as being ``"top_down"``, or if no additional
    information is provided, we assume that the upwards direction matches that
    of the vector ``[0, 0, -1]``. If the camera view is ``"bottom_up"``, the
    upwards direction is assumed to be given by ``[0, 0, 1]``. For both cases,
    we assume that position values are expressed in the image coordinate
    system (where the positive X-axis is oriented to the right, the positive
    Y-axis faces downwards, and positive Z-axis faces away from the person
    viewing the screen).

    If one of the required pieces of information is missing for a frame (e.g.,
    the left keypoint is not visible), then the computed head direction vector
    is set to NaN.

    """
    # Validate input data
    _validate_type_data_array(data)
    validate_dims_coords(
        data,
        {
            "time": [],
            "keypoint": [left_keypoint, right_keypoint],
            "space": [],
        },
    )
    if len(data.space) != 2:
        raise logger.error(
            ValueError(
                "Input data must have exactly 2 spatial dimensions, but "
                f"currently has {len(data.space)}."
            )
        )
    # Validate input keypoints
    if left_keypoint == right_keypoint:
        raise logger.error(
            ValueError("The left and right keypoints may not be identical.")
        )
    # Define right-to-left vector
    right_to_left_vector = data.sel(
        keypoint=left_keypoint, drop=True
    ) - data.sel(keypoint=right_keypoint, drop=True)
    # Define upward vector
    # default: negative z direction in the image coordinate system
    upward_vector_arr = (
        np.array([0, 0, -1])
        if camera_view == "top_down"
        else np.array([0, 0, 1])
    )
    upward_vector = xr.DataArray(
        np.tile(upward_vector_arr.reshape(1, -1), [len(data.time), 1]),
        dims=["time", "space"],
        coords={
            "space": ["x", "y", "z"],
        },
    )
    # Compute forward direction as the cross product
    # (right-to-left) cross (forward) = up
    forward_vector = cast(
        "xr.DataArray",
        xr.cross(right_to_left_vector, upward_vector, dim="space"),
    ).drop_sel(
        space="z"
    )  # keep only the first 2 spatal dimensions of the result
    # Return unit vector
    result = convert_to_unit(forward_vector)
    result.name = "forward_vector"
    return result


def compute_head_direction_vector(
    data: xr.DataArray,
    left_keypoint: str,
    right_keypoint: str,
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
) -> xr.DataArray:
    """Compute the 2D head direction vector given two keypoints on the head.

    This function is an alias for :func:`compute_forward_vector()\
    <movement.kinematics.compute_forward_vector>`. For more
    detailed information on how the head direction vector is computed,
    please refer to the documentation for that function.

    Parameters
    ----------
    data
        The input data representing position. This must contain
        the two chosen keypoints corresponding to the left and
        right of the head.
    left_keypoint
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint
        Name of the right keypoint, e.g., "right_ear"
    camera_view
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the head direction vector, with
        dimensions matching the input data array, but without the
        ``keypoint`` dimension.

    """
    result = compute_forward_vector(
        data, left_keypoint, right_keypoint, camera_view=camera_view
    )
    result.name = "head_direction_vector"
    return result


def compute_forward_vector_angle(
    data: xr.DataArray,
    left_keypoint: Hashable,
    right_keypoint: Hashable,
    reference_vector: xr.DataArray | ArrayLike = (1, 0),
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
    in_degrees: bool = False,
) -> xr.DataArray:
    r"""Compute the signed angle between a reference and a forward vector.

    Forward vector angle is the :func:`signed angle\
    <movement.utils.vector.compute_signed_angle_2d>`
    between the reference vector and the animal's :func:`forward vector\
    <movement.kinematics.compute_forward_vector>`.
    The returned angles are in radians, spanning the range :math:`(-\pi, \pi]`,
    unless ``in_degrees`` is set to ``True``.

    Parameters
    ----------
    data
        The input data representing position. This must contain
        the two symmetrical keypoints located on the left and
        right sides of the body, respectively.
    left_keypoint
        Name of the left keypoint, e.g., "left_ear", used to compute the
        forward vector.
    right_keypoint
        Name of the right keypoint, e.g., "right_ear", used to compute the
        forward vector.
    reference_vector
        The reference vector against which the ``forward_vector`` is
        compared to compute 2D heading. Must be a two-dimensional vector,
        in the form [x,y] - where ``reference_vector[0]`` corresponds to the
        x-coordinate and ``reference_vector[1]`` corresponds to the
        y-coordinate. If left unspecified, the vector [1, 0] is used by
        default.
    camera_view
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.
    in_degrees
        If ``True``, the returned heading array is given in degrees.
        Otherwise, the array is given in radians. Default ``False``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed forward vector angles,
        with dimensions matching the input data array,
        but without the ``keypoint`` and ``space`` dimensions.

    See Also
    --------
    movement.utils.vector.compute_signed_angle_2d :
        The underlying function used to compute the signed angle between two
        2D vectors. See this function for a definition of the signed
        angle between two vectors.
    movement.kinematics.compute_forward_vector :
        The function used to compute the forward vector.

    """
    # Convert reference vector to np.array if not already a valid array
    if not isinstance(reference_vector, np.ndarray | xr.DataArray):
        reference_vector = np.array(reference_vector)

    # Compute forward vector
    forward_vector = compute_forward_vector(
        data, left_keypoint, right_keypoint, camera_view=camera_view
    )

    # Compute signed angle between reference vector and forward vector
    heading_array = compute_signed_angle_2d(
        forward_vector, reference_vector, v_as_left_operand=True
    )

    # Convert to degrees
    if in_degrees:
        heading_array = cast("xr.DataArray", np.rad2deg(heading_array))

    heading_array.name = "forward_vector_angle"
    return heading_array


def compute_angular_time_derivative(
    data: xr.DataArray,
    order: int,
    window: int | None = None,
) -> xr.DataArray:
    r"""Compute the time-derivative of a 2D orientation.

    The orientation may be given either as 2D vectors or as angles in
    radians. It is first converted into a continuous (unwrapped) angle,
    which avoids the spurious jumps that occur when angles wrap around
    at :math:`\pm\pi` (see Notes). The unwrapped angle is then
    differentiated with respect to ``time``.

    Parameters
    ----------
    data
        The input orientation data, containing ``time`` as a dimension.
        If it contains a ``space`` dimension, it must have exactly the
        coordinates ``["x", "y"]`` and is interpreted as 2D vectors, which
        need not be of unit length. Otherwise, it is interpreted as angles
        in radians.
    order
        The order of the time-derivative. Use 1 for angular velocity and
        2 for angular acceleration. Must be a positive integer.
    window
        If ``None`` (default), the derivative is computed with
        second-order accurate central differences, as in
        :func:`movement.kinematics.compute_time_derivative`. If an
        integer, the derivative is instead estimated by a least-squares
        polynomial fit over a centred window of ``window`` frames (see
        Notes).

    Returns
    -------
    xarray.DataArray
        The time-derivative of the orientation, in radians per unit of
        time (raised to the power of ``order``). It has the dimensions of
        the input, without ``space``.

    Notes
    -----
    Naively differentiating angles produces large spurious spikes
    whenever the angle wraps around from :math:`\pi` to :math:`-\pi`
    (or vice versa). To avoid this, we compute the
    :func:`signed angle <movement.utils.vector.compute_signed_angle_2d>`
    between consecutive orientations. Each increment lies in
    :math:`(-\pi, \pi]`, so wrap-arounds have no effect, and the
    cumulative sum of the increments gives a continuous angle. Gaps
    (NaNs or zero-length vectors) are bridged by the rotation between
    the last valid frame before the gap and the first valid frame after
    it.

    This assumes that the orientation rotates by less than :math:`\pi`
    between consecutive valid frames. Larger rotations are
    indistinguishable from smaller rotations in the opposite direction.
    They typically indicate tracking errors, such as swapped left/right
    keypoints.

    Positive values correspond to rotations from the positive x-axis
    towards the positive y-axis. In image coordinates (y pointing down)
    with a top-down camera, this is a clockwise rotation on screen,
    i.e. a right turn of the animal.

    To smooth the orientation *before* differentiating, smooth the
    vectors rather than the angles, for example with
    :func:`movement.filtering.rolling_filter`. A rolling mean of the
    vector components is a circular mean, whereas a rolling mean of
    angles is wrong near :math:`\pm\pi`. Likewise, fill gaps with
    :func:`movement.filtering.interpolate_over_time` applied to the
    vectors, not to the angles.

    See Also
    --------
    compute_angular_velocity : Wrapper for ``order=1``.
    movement.kinematics.compute_time_derivative :
        The time-derivative of non-circular data.

    """
    _validate_type_data_array(data)
    if not isinstance(order, int) or order <= 0:
        raise logger.error(
            ValueError(f"Order must be a positive integer, but got {order}.")
        )
    validate_dims_coords(data, {"time": []})
    theta = _unwrap_orientation(data)
    return compute_time_derivative(theta, order)


def compute_angular_velocity(
    data: xr.DataArray,
    window: int | None = None,
    in_degrees: bool = False,
) -> xr.DataArray:
    r"""Compute the angular velocity of a 2D orientation.

    The orientation may be given either as 2D vectors (e.g. the output
    of :func:`movement.kinematics.compute_forward_vector`) or as angles
    in radians (e.g. the output of
    :func:`movement.kinematics.compute_forward_vector_angle`). Angle
    wrap-around at :math:`\pm\pi` is handled. See
    :func:`compute_angular_time_derivative` for details.

    Parameters
    ----------
    data
        The input orientation data, containing ``time`` as a dimension.
        If it contains a ``space`` dimension, it must have exactly the
        coordinates ``["x", "y"]`` and is interpreted as 2D vectors.
        Otherwise, it is interpreted as angles in radians.
    window
        If ``None`` (default), the unsmoothed angular velocity is
        computed with central differences. If an integer, it is the
        slope of a least-squares line fitted to the unwrapped angle over
        a centred window of ``window`` frames.
    in_degrees
        If ``True``, the output is in degrees per unit of time.
        Otherwise (default), it is in radians per unit of time.

    Returns
    -------
    xarray.DataArray
        The angular velocity, with the dimensions of the input, without
        ``space``. The unit of time is that of the ``time`` coordinate
        (seconds or frames).

    See Also
    --------
    compute_angular_time_derivative : The underlying function used.

    """
    result = compute_angular_time_derivative(data, order=1, window=window)
    if in_degrees:
        result = cast("xr.DataArray", np.rad2deg(result))
    result.name = "angular_velocity"
    return result


def _unwrap_orientation(data: xr.DataArray) -> xr.DataArray:
    """Convert orientation vectors or angles to a continuous angle.

    The result is in radians, with an arbitrary constant offset, and is
    NaN wherever the orientation is undefined (NaN or zero-length).
    """
    if "space" in data.dims:
        validate_dims_coords(data, {"space": ["x", "y"]}, exact_coords=True)
        vectors = data
    else:
        vectors = xr.concat(
            [cast("xr.DataArray", f(data)) for f in (np.cos, np.sin)],
            dim="space",
        ).assign_coords(space=["x", "y"])
    valid = compute_norm(vectors) > 0  # False for NaN and null vectors
    filled = vectors.where(valid).ffill(dim="time")
    increments = compute_signed_angle_2d(filled.shift(time=1), filled)
    return increments.fillna(0).cumsum(dim="time").where(valid)


def _validate_type_data_array(data: xr.DataArray) -> None:
    """Validate the input data is an xarray DataArray.

    Parameters
    ----------
    data
        The input data to validate.

    Raises
    ------
    ValueError
        If the input data is not an xarray DataArray.

    """
    if not isinstance(data, xr.DataArray):
        raise logger.error(
            TypeError(
                "Input data must be an xarray.DataArray, "
                f"but got {type(data)}."
            )
        )
