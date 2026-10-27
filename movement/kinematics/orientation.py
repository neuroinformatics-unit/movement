"""Compute orientations as vectors and angles."""

from collections.abc import Hashable
from typing import Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy.signal import savgol_coeffs

from movement.kinematics.kinematics import compute_time_derivative
from movement.utils.logging import logger
from movement.utils.vector import (
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

    Raises
    ------
    ValueError
        If ``camera_view`` is not one of ``"top_down"`` or
        ``"bottom_up"``.

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
    # Validate camera view
    if camera_view not in ("top_down", "bottom_up"):
        raise logger.error(
            ValueError(
                "camera_view must be 'top_down' or 'bottom_up', "
                f"but got {camera_view!r}."
            )
        )
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

    The orientation is given as 2D vectors. Its rate of change is
    estimated from the signed angles between nearby orientation vectors,
    which avoids the spurious jumps that occur when angles wrap around at
    :math:`\pm\pi` (see Notes).

    Parameters
    ----------
    data
        The input orientation vectors, containing ``time`` and ``space``
        as dimensions. ``space`` must have exactly the coordinates
        ``["x", "y"]``. The vectors need not be of unit length.
    order
        The order of the time-derivative. Use 1 for angular velocity and
        2 for angular acceleration. Must be a positive integer.
    window
        If ``None`` (default), the derivative is computed with central
        differences (one-sided at the first and last observations).
        Otherwise, the size of the centred rolling window, representing
        the fixed number of observations to which a least-squares
        polynomial is fitted (see Notes). Must be an odd integer greater
        than ``order``, and requires a uniformly sampled ``time``
        coordinate.

    Returns
    -------
    xarray.DataArray
        The time-derivative of the orientation, in radians per unit of
        time (raised to the power of ``order``). It has the dimensions of
        the input, without ``space``.

    Notes
    -----
    **Wrap-around.** Differentiating angles directly produces spurious
    spikes wherever they wrap around at :math:`\pm\pi`.
    Instead, this function uses the
    :func:`signed angles<movement.utils.vector.compute_signed_angle_2d>`
    between nearby orientations vectors. Without
    ``window``, ``order=1`` is a central-difference estimate built from
    the angles between consecutive observations. As with
    :func:`numpy.gradient`, it stays centred on each observation even
    when time steps are irregular. Higher orders differentiate that
    result further with
    :func:`~movement.kinematics.compute_time_derivative`. With ``window``,
    the angles are measured from the window's centre orientation (see
    *Windowed estimate*). This assumes rotations of less than :math:`\pi`
    between consecutive observations (or across half a ``window``).
    Undefined orientations (NaNs or zero-length vectors) are treated as
    NaN.

    **Sign.** Positive values are rotations from the positive x-axis
    towards the positive y-axis. In image coordinates (y pointing down)
    with a top-down camera, this is clockwise on screen.

    **Windowed estimate.** With ``window``, the angles of the
    orientations in the window, relative to the centre one, are
    differentiated with Savitzky-Golay coefficients
    (:func:`scipy.signal.savgol_coeffs`). For ``order=1``, this is the
    slope of a least-squares line over the window centred on each time
    point. It uses every observation in the window. The first and last
    ``window // 2`` values are NaN, as are all outputs whose window contains
    a NaN. For a window *ending* on each time point, shift the result by
    ``window // 2`` time steps.

    **Preprocessing.** Smooth, interpolate, or resample the orientation
    *vectors*, not their angles, since angle averages are wrong near
    :math:`\pm\pi`. Windowed estimates require uniform sampling, so make sure
    to resample irregularly sampled data onto a uniform ``time`` grid first,
    e.g. with :meth:`xarray.DataArray.interp` on the vectors.

    See Also
    --------
    compute_angular_velocity : Wrapper for ``order=1``.
    compute_angular_acceleration : Wrapper for ``order=2``.

    """
    _validate_type_data_array(data)
    if not isinstance(order, int) or order <= 0:
        raise logger.error(
            ValueError(f"Order must be a positive integer, but got {order}.")
        )
    if window is not None and not (
        isinstance(window, int) and window > order and window % 2 == 1
    ):
        raise logger.error(
            ValueError(
                f"Window must be an odd integer greater than the order "
                f"({order}), but got {window}."
            )
        )
    validate_dims_coords(
        data, {"time": [], "space": ["x", "y"]}, exact_coords=True
    )
    if window is None:
        # Mean angular velocity over each time step (divided along "time"
        # by name), interpolated from the step midpoints to the time
        # points. The first and last steps' rates are used at the edges.
        vectors = data.drop_vars("time")
        step_angle = compute_signed_angle_2d(
            vectors.isel(time=slice(None, -1)),
            vectors.isel(time=slice(1, None)),
        )
        time = data["time"].values
        step_dt = xr.DataArray(np.diff(time), dims="time")
        midpoints = (time[:-1] + time[1:]) / 2
        result = (
            (step_angle / step_dt)
            .pad(time=1, mode="edge")
            .assign_coords(time=np.r_[time[0], midpoints, time[-1]])
            .interp(time=time)
        )
        # The result is a rate, not an angle, so it doesn't wrap around;
        # higher orders can use the ordinary time derivative.
        if order > 1:
            result = compute_time_derivative(result, order - 1)
    else:
        coeffs = savgol_coeffs(
            window,
            order,
            deriv=order,
            delta=_uniform_time_step(data),
            use="dot",
        )
        # Out-of-range neighbours are NaN, so the edge outputs are NaN
        neighbours = data.rolling(time=window, center=True).construct("offset")
        relative_angle = compute_signed_angle_2d(data, neighbours)
        result = (relative_angle * xr.DataArray(coeffs, dims="offset")).sum(
            "offset", skipna=False
        )
    # Drop attrs for consistency with compute_time_derivative (for now).
    # See https://github.com/neuroinformatics-unit/movement/issues/1158
    result.attrs = {}
    result.name = "angular_time_derivative"
    return result


def compute_angular_velocity(
    data: xr.DataArray,
    window: int | None = None,
    in_degrees: bool = False,
) -> xr.DataArray:
    r"""Compute the angular velocity of a 2D orientation.

    The orientation is given as 2D vectors, e.g. the output of
    :func:`compute_forward_vector`.

    Parameters
    ----------
    data
        The input orientation vectors, containing ``time`` and ``space``
        as dimensions. ``space`` must have exactly the coordinates
        ``["x", "y"]``. The vectors need not be of unit length.
    window
        If ``None`` (default), the angular velocity is computed with
        central differences (one-sided at the first and last
        observations). Otherwise, the size of the centred rolling window,
        representing the fixed number of observations to which a
        least-squares line is fitted; the angular velocity is its slope.
        Must be an odd integer greater than 1, and requires a uniformly
        sampled ``time`` coordinate.
    in_degrees
        If ``False`` (default), the output is in radians per unit of
        time. If ``True``, it is in degrees per unit of time.

    Returns
    -------
    xarray.DataArray
        The angular velocity, with the dimensions of the input, without
        ``space``. The unit of time is that of the ``time`` coordinate
        (seconds or frames).

    See Also
    --------
    compute_angular_time_derivative : The underlying function used.
    compute_angular_acceleration : The second time-derivative of orientation.

    Examples
    --------
    >>> from movement.kinematics import (
    ...     compute_angular_velocity,
    ...     compute_forward_vector,
    ... )
    >>> head_vector = compute_forward_vector(
    ...     ds.position, "left_ear", "right_ear"
    ... )

    Central-difference angular head velocity:

    >>> ahv = compute_angular_velocity(head_vector)

    Least-squares estimate over the 7 observations centred on each time
    point, in degrees per unit of time:

    >>> ahv = compute_angular_velocity(head_vector, window=7, in_degrees=True)

    Shift the above estimate by half a window to get a trailing
    estimate, using only the 7 observations up to and including each
    time point, in radians per unit of time:

    >>> ahv_trailing = compute_angular_velocity(head_vector, window=7).shift(
    ...     time=7 // 2
    ... )

    """
    result = compute_angular_time_derivative(data, order=1, window=window)
    if in_degrees:
        result = cast("xr.DataArray", np.rad2deg(result))
    result.name = "angular_velocity"
    return result


def compute_angular_acceleration(
    data: xr.DataArray,
    window: int | None = None,
    in_degrees: bool = False,
) -> xr.DataArray:
    r"""Compute the angular acceleration of a 2D orientation.

    The orientation is given as 2D vectors, e.g. the output of
    :func:`compute_forward_vector`.

    Parameters
    ----------
    data
        The input orientation vectors, containing ``time`` and ``space``
        as dimensions. ``space`` must have exactly the coordinates
        ``["x", "y"]``. The vectors need not be of unit length.
    window
        If ``None`` (default), the angular acceleration is computed by
        differentiating the central-difference angular velocity.
        Otherwise, the size of the centred rolling window, representing
        the fixed number of observations to which a least-squares
        quadratic is fitted; the angular acceleration is its second
        derivative. Must be an odd integer greater than 2, and requires
        a uniformly sampled ``time`` coordinate.
    in_degrees
        If ``False`` (default), the output is in radians per unit of
        time squared. If ``True``, it is in degrees per unit of time
        squared.

    Returns
    -------
    xarray.DataArray
        The angular acceleration, with the dimensions of the input,
        without ``space``. The unit of time is that of the ``time``
        coordinate (seconds or frames).

    See Also
    --------
    compute_angular_time_derivative : The underlying function used.
    compute_angular_velocity : The first time-derivative of orientation.

    Examples
    --------
    >>> from movement.kinematics import (
    ...     compute_angular_acceleration,
    ...     compute_forward_vector,
    ... )
    >>> head_vector = compute_forward_vector(
    ...     ds.position, "left_ear", "right_ear"
    ... )

    Least-squares estimate over the 9 observations centred on each time
    point, in degrees per unit of time squared:

    >>> aha = compute_angular_acceleration(
    ...     head_vector, window=9, in_degrees=True
    ... )

    """
    result = compute_angular_time_derivative(data, order=2, window=window)
    if in_degrees:
        result = cast("xr.DataArray", np.rad2deg(result))
    result.name = "angular_acceleration"
    return result


def _uniform_time_step(data: xr.DataArray) -> float:
    """Return the time step, which must be uniform."""
    steps = np.diff(data["time"].values)
    if not np.allclose(steps, steps[0]):
        raise logger.error(
            ValueError(
                "A uniformly sampled 'time' coordinate is required "
                "when 'window' is set."
            )
        )
    return float(steps[0])


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
