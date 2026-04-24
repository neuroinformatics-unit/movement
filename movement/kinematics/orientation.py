"""Compute orientations as vectors and angles."""

from collections.abc import Hashable
from typing import Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.kinematics.kinematics import compute_backward_displacement
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
    """Compute a 2D forward vector given two left-right symmetric keypoint.

    The forward vector is computed as a vector perpendicular to the
    line connecting two symmetrical keypoint on either side of the body
    (i.e., symmetrical relative to the mid-sagittal plane), and pointing
    forwards (in the rostral direction). A top-down or bottom-up view of the
    animal is assumed (see Notes).

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two symmetrical keypoint located on the left and
        right sides of the body, respectively.
    left_keypoint : Hashable
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint : Hashable
        Name of the right keypoint, e.g., "right_ear"
    camera_view : Literal["top_down", "bottom_up"], optional
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
    keypoint. The upwards direction, in turn, can be determined by passing the
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
    # Validate input keypoint
    if left_keypoint == right_keypoint:
        raise logger.error(
            ValueError("The left and right keypoint may not be identical.")
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
):
    """Compute the 2D head direction vector given two keypoint on the head.

    This function is an alias for :func:`compute_forward_vector()\
    <movement.kinematics.compute_forward_vector>`. For more
    detailed information on how the head direction vector is computed,
    please refer to the documentation for that function.

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two chosen keypoint corresponding to the left and
        right of the head.
    left_keypoint : str
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint : str
        Name of the right keypoint, e.g., "right_ear"
    camera_view : Literal["top_down", "bottom_up"], optional
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
    data : xarray.DataArray
        The input data representing position. This must contain
        the two symmetrical keypoint located on the left and
        right sides of the body, respectively.
    left_keypoint : Hashable
        Name of the left keypoint, e.g., "left_ear", used to compute the
        forward vector.
    right_keypoint : Hashable
        Name of the right keypoint, e.g., "right_ear", used to compute the
        forward vector.
    reference_vector : xr.DataArray | ArrayLike, optional
        The reference vector against which the ``forward_vector`` is
        compared to compute 2D heading. Must be a two-dimensional vector,
        in the form [x,y] - where ``reference_vector[0]`` corresponds to the
        x-coordinate and ``reference_vector[1]`` corresponds to the
        y-coordinate. If left unspecified, the vector [1, 0] is used by
        default.
    camera_view : Literal["top_down", "bottom_up"], optional
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.
    in_degrees : bool
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
        raise logger.error(
            TypeError(
                "Input data must be an xarray.DataArray, "
                f"but got {type(data)}."
            )
        )


def compute_turning_angle(
    data: xr.DataArray,
    in_degrees: bool = False,
    min_step_length: float = 0.0,
) -> xr.DataArray:
    r"""Compute the turning angles between consecutive steps in a trajectory.

    The turning angle at time ``t`` is  the :func:`signed angle\
    <movement.utils.vector.compute_signed_angle_2d>` between two
    consecutive :func:`backward displacement\
    <movement.kinematics.compute_backward_displacement>` vectors
    at times ``t-1`` and ``t``.
    The returned angles are in radians, spanning the range :math:`(-\pi, \pi]`,
    unless ``in_degrees`` is set to ``True``.

    Parameters
    ----------
    data
        The input position data. Must contain ``time`` and ``space``
        dimensions. The ``space`` dimension must contain exactly the
        coordinates ``["x", "y"]`` (2D spatial data only).
    in_degrees
        If ``True``, return turning angles in degrees. Default is
        ``False`` (radians).
    min_step_length
        The minimum step length to consider for computing the turning
        angle. Any turning angle involving an incoming or outgoing step
        shorter than or equal to this value is set to ``NaN``. The
        default ``0.0`` only masks steps with exactly zero length,
        which means steps with near-zero lengths may still produce
        spurious angles. See Note 2 below.

    Returns
    -------
    xr.DataArray
        Turning angles with the same shape as the input ``data``, but
        with the ``space`` dimension dropped.

    Notes
    -----
    1. **Time dimension length:** This function uses a ``shift``
       operation to preserve the original ``time`` dimension length.
       The first two time steps are always ``NaN``: the first because
       no previous step exists, and the second because a turning angle
       requires two steps (three positions). In other words, the turning angle
       at time step ``t`` is computed as the angle between the steps
       from ``t-2`` to ``t-1`` and from ``t-1`` to ``t``.
    2. **Positional jitter and small steps:** Tracking data
       often contains positional jitter, meaning a stationary animal
       may appear to make microscopic movements. With default parameters
       (``min_step_length=0.0``), these tiny, noisy movements will
       produce spurious, meaningless turning angles. It is highly
       recommended to set ``min_step_length`` to an appropriate threshold
       based on the tracking resolution and the animal's size in the scene.
       The value should be in the same units as the input position data
       (e.g. pixels, mm, etc.). Pre-smoothing the trajectory
       can also help reduce positional jitter.
    3. **NaN propagation:** ``NaN`` positions in the input propagate
       to ``NaN`` turning angles. A single missing position affects
       up to two turning angles (the incoming and outgoing steps).
       Use :func:`movement.filtering.interpolate_over_time` to fill
       positional gaps before computing turning angles if continuity
       is important.

    See Also
    --------
    movement.kinematics.compute_backward_displacement :
        The underlying function used to compute the displacement vectors.
    movement.utils.vector.compute_signed_angle_2d :
        The underlying function used to compute the signed angle
        between two consecutive displacement vectors.

    Examples
    --------
    >>> from movement.kinematics import compute_turning_angle

    Compute turning angles from the centroid trajectory of a poses
    dataset ``ds``:

    >>> centroid = ds.position.mean(dim="keypoints")
    >>> angles = compute_turning_angle(centroid)

    Compute in degrees, with a minimum step length of 3 pixels to filter out
    pose estimation jitter:

    >>> angles = compute_turning_angle(
    ...     centroid, in_degrees=True, min_step_length=3
    ... )

    """
    validate_dims_coords(
        data, {"time": [], "space": ["x", "y"]}, exact_coords=True
    )

    # Displacement arriving at each time step t.
    disp = compute_backward_displacement(data)

    # Turning angle at t = rotation needed to align step[t-1] onto step[t].
    turning = compute_signed_angle_2d(disp.shift(time=1), disp)

    # Mask turning angles involving steps smaller than min_step_length
    step_lengths = compute_norm(disp)
    invalid_steps = (step_lengths <= min_step_length) | (
        step_lengths.shift(time=1) <= min_step_length
    )
    turning = xr.where(invalid_steps, np.nan, turning)

    turning.attrs["units"] = "radians"

    if in_degrees:
        turning = np.rad2deg(turning)
        turning.attrs["units"] = "degrees"

    turning.name = "turning_angle"

    return turning
