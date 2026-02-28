"""Compute orientations as vectors and angles."""

import warnings
from collections.abc import Hashable
from typing import Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.utils.logging import logger
from movement.utils.vector import (
    compute_signed_angle_2d,
    convert_to_unit,
)
from movement.validators.arrays import validate_dims_coords


def compute_perpendicular_vector(
    data: xr.DataArray,
    left_keypoint: Hashable,
    right_keypoint: Hashable,
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
) -> xr.DataArray:
    """Compute a 2D perpendicular vector given two symmetric keypoints.

    The perpendicular vector is computed as a vector perpendicular
    to the line connecting two symmetrical keypoints on either side
    of the body (i.e., symmetrical relative to the mid-sagittal
    plane), and pointing forwards (in the rostral direction). A
    top-down or bottom-up view of the animal is assumed (see Notes).

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two symmetrical keypoints located on the left and
        right sides of the body, respectively.
    left_keypoint : Hashable
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint : Hashable
        Name of the right keypoint, e.g., "right_ear"
    camera_view : Literal["top_down", "bottom_up"], optional
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"``
        (where the upwards direction is [0, 0, -1]), or
        ``"bottom_up"`` (where the upwards direction is
        [0, 0, 1]). If left unspecified, the camera view is
        assumed to be ``"top_down"``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the perpendicular
        vector, with dimensions matching the input data array,
        but without the ``keypoints`` dimension.

    Notes
    -----
    To determine the forward direction of the animal, we need to
    specify (1) the right-to-left direction of the animal and
    (2) its upward direction. We determine the right-to-left
    direction via the input left and right keypoints. The upwards
    direction, in turn, can be determined by passing the
    ``camera_view`` argument with either ``"top_down"`` or
    ``"bottom_up"``. If the camera view is specified as being
    ``"top_down"``, or if no additional information is provided,
    we assume that the upwards direction matches that of the
    vector ``[0, 0, -1]``. If the camera view is ``"bottom_up"``,
    the upwards direction is assumed to be given by
    ``[0, 0, 1]``. For both cases, we assume that position values
    are expressed in the image coordinate system (where the
    positive X-axis is oriented to the right, the positive Y-axis
    faces downwards, and positive Z-axis faces away from the
    person viewing the screen).

    If one of the required pieces of information is missing for a
    frame (e.g., the left keypoint is not visible), then the
    computed perpendicular vector is set to NaN.

    See Also
    --------
    compute_vector_from_to : Compute a vector from one keypoint
        to another by subtracting position vectors.

    """
    # Validate input data
    _validate_type_data_array(data)
    validate_dims_coords(
        data,
        {
            "time": [],
            "keypoints": [left_keypoint, right_keypoint],
            "space": [],
        },
    )
    if len(data.space) != 2:
        raise logger.error(
            ValueError(
                "Input data must have exactly 2 spatial "
                "dimensions, but currently has "
                f"{len(data.space)}."
            )
        )
    # Validate input keypoints
    if left_keypoint == right_keypoint:
        raise logger.error(
            ValueError(
                "The left and right keypoints may not be "
                "identical."
            )
        )
    # Define right-to-left vector
    right_to_left_vector = data.sel(
        keypoints=left_keypoint, drop=True
    ) - data.sel(keypoints=right_keypoint, drop=True)
    # Define upward vector
    # default: negative z direction in the image coordinate system
    upward_vector_arr = (
        np.array([0, 0, -1])
        if camera_view == "top_down"
        else np.array([0, 0, 1])
    )
    upward_vector = xr.DataArray(
        np.tile(
            upward_vector_arr.reshape(1, -1),
            [len(data.time), 1],
        ),
        dims=["time", "space"],
        coords={
            "space": ["x", "y", "z"],
        },
    )
    # Compute forward direction as the cross product
    # (right-to-left) cross (forward) = up
    forward_vector = cast(
        "xr.DataArray",
        xr.cross(
            right_to_left_vector, upward_vector, dim="space"
        ),
    ).drop_sel(
        space="z"
    )  # keep only the first 2 spatial dimensions
    # Return unit vector
    result = convert_to_unit(forward_vector)
    result.name = "perpendicular_vector"
    return result


def compute_vector_from_to(
    data: xr.DataArray,
    from_keypoint: Hashable,
    to_keypoint: Hashable,
) -> xr.DataArray:
    """Compute a unit vector from one keypoint to another.

    The vector is computed by subtracting the position of the
    ``from_keypoint`` from the position of the ``to_keypoint``,
    and then normalising to a unit vector.

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two keypoints specified by ``from_keypoint`` and
        ``to_keypoint``.
    from_keypoint : Hashable
        Name of the origin keypoint, e.g., "neck" or "tail_base"
    to_keypoint : Hashable
        Name of the target keypoint, e.g., "nose" or "head"

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the unit direction
        vector from ``from_keypoint`` to ``to_keypoint``, with
        dimensions matching the input data array, but without
        the ``keypoints`` dimension.

    Notes
    -----
    If one of the required keypoints is missing (NaN) for a
    frame, the computed vector is set to NaN for that frame.

    See Also
    --------
    compute_perpendicular_vector : Compute a 2D perpendicular
        vector given two left-right symmetric keypoints.

    Examples
    --------
    Compute the back-to-front direction vector using neck and
    nose keypoints:

    >>> vector = compute_vector_from_to(
    ...     ds.position, from_keypoint="neck", to_keypoint="nose"
    ... )

    """
    # Validate input data
    _validate_type_data_array(data)
    validate_dims_coords(
        data,
        {
            "time": [],
            "keypoints": [from_keypoint, to_keypoint],
            "space": [],
        },
    )
    if from_keypoint == to_keypoint:
        raise logger.error(
            ValueError(
                "The from_keypoint and to_keypoint may not be "
                "identical."
            )
        )
    # Compute direction vector
    direction_vector = data.sel(
        keypoints=to_keypoint, drop=True
    ) - data.sel(keypoints=from_keypoint, drop=True)
    # Return unit vector
    result = convert_to_unit(direction_vector)
    result.name = "vector_from_to"
    return result


def compute_vector_angle(
    vector: xr.DataArray,
    reference_vector: xr.DataArray | ArrayLike = (1, 0),
    in_degrees: bool = False,
) -> xr.DataArray:
    r"""Compute the signed angle of a vector relative to a reference.

    The angle is the :func:`signed angle\
    <movement.utils.vector.compute_signed_angle_2d>` from the
    reference vector to the input vector. The returned angles are
    in radians, spanning the range :math:`(-\pi, \pi]`, unless
    ``in_degrees`` is set to ``True``.

    Parameters
    ----------
    vector : xarray.DataArray
        A 2D vector (or array of 2D vectors) for which to
        compute the angle. Must contain the ``space`` dimension
        with ``"x"`` and ``"y"`` coordinates.
    reference_vector : xr.DataArray | ArrayLike, optional
        The reference vector against which the angle is computed.
        Must be a two-dimensional vector in the form [x, y].
        Can be an ``xr.DataArray`` with a ``time`` axis for
        time-varying references. If left unspecified, the vector
        ``[1, 0]`` (positive x-axis) is used by default.
    in_degrees : bool
        If ``True``, the returned angles are given in degrees.
        Otherwise, the angles are given in radians.
        Default ``False``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed angles, with
        dimensions matching the input ``vector`` but without the
        ``space`` dimension.

    See Also
    --------
    movement.utils.vector.compute_signed_angle_2d :
        The underlying function used to compute the signed angle
        between two 2D vectors.
    compute_perpendicular_vector :
        Compute a 2D perpendicular vector from left-right
        keypoints, which can be passed to this function.
    compute_vector_from_to :
        Compute a direction vector between two keypoints, which
        can be passed to this function.

    Examples
    --------
    Compute the angle of a forward vector relative to the x-axis:

    >>> fwd = compute_perpendicular_vector(
    ...     ds.position, "left_ear", "right_ear"
    ... )
    >>> angle = compute_vector_angle(fwd)

    Compute the angle using a time-varying reference:

    >>> angle = compute_vector_angle(fwd, reference_vector=ref)

    """
    _validate_type_data_array(vector)
    validate_dims_coords(
        vector, {"space": ["x", "y"]}, exact_coords=True
    )

    # Convert reference vector to np.array if needed
    if not isinstance(
        reference_vector, np.ndarray | xr.DataArray
    ):
        reference_vector = np.array(reference_vector)

    # Compute signed angle between reference and input vector
    angle_array = compute_signed_angle_2d(
        vector, reference_vector, v_as_left_operand=True
    )

    # Convert to degrees
    if in_degrees:
        angle_array = cast(
            "xr.DataArray", np.rad2deg(angle_array)
        )

    angle_array.name = "vector_angle"
    return angle_array


def compute_head_direction_vector(
    data: xr.DataArray,
    left_keypoint: str,
    right_keypoint: str,
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
):
    """Compute the 2D head direction vector given two keypoints.

    This function is an alias for
    :func:`compute_perpendicular_vector`. For more detailed
    information on how the head direction vector is computed,
    please refer to the documentation for that function.

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two chosen keypoints corresponding to the left and
        right of the head.
    left_keypoint : str
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint : str
        Name of the right keypoint, e.g., "right_ear"
    camera_view : Literal["top_down", "bottom_up"], optional
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"``
        (where the upwards direction is [0, 0, -1]), or
        ``"bottom_up"`` (where the upwards direction is
        [0, 0, 1]). If left unspecified, the camera view is
        assumed to be ``"top_down"``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the head direction
        vector, with dimensions matching the input data array,
        but without the ``keypoints`` dimension.

    """
    result = compute_perpendicular_vector(
        data,
        left_keypoint,
        right_keypoint,
        camera_view=camera_view,
    )
    result.name = "head_direction_vector"
    return result


def compute_forward_vector(
    data: xr.DataArray,
    left_keypoint: Hashable,
    right_keypoint: Hashable,
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
) -> xr.DataArray:
    """Compute a 2D forward vector given two symmetric keypoints.

    .. deprecated::
        Use :func:`compute_perpendicular_vector` instead.

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position.
    left_keypoint : Hashable
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint : Hashable
        Name of the right keypoint, e.g., "right_ear"
    camera_view : Literal["top_down", "bottom_up"], optional
        The camera viewing angle. Default ``"top_down"``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the forward vector.

    """
    warnings.warn(
        "`compute_forward_vector` is deprecated and will be "
        "removed in a future release. Use "
        "`compute_perpendicular_vector` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = compute_perpendicular_vector(
        data,
        left_keypoint,
        right_keypoint,
        camera_view=camera_view,
    )
    result.name = "forward_vector"
    return result


def compute_forward_vector_angle(
    data: xr.DataArray,
    left_keypoint: Hashable,
    right_keypoint: Hashable,
    reference_vector: xr.DataArray | ArrayLike = (1, 0),
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
    in_degrees: bool = False,
) -> xr.DataArray:
    r"""Compute the signed angle of a forward vector.

    .. deprecated::
        Use :func:`compute_perpendicular_vector` followed by
        :func:`compute_vector_angle` instead.

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position.
    left_keypoint : Hashable
        Name of the left keypoint.
    right_keypoint : Hashable
        Name of the right keypoint.
    reference_vector : xr.DataArray | ArrayLike, optional
        The reference vector. Default ``(1, 0)``.
    camera_view : Literal["top_down", "bottom_up"], optional
        The camera viewing angle. Default ``"top_down"``.
    in_degrees : bool
        If ``True``, return degrees. Default ``False``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the forward vector angles.

    """
    warnings.warn(
        "`compute_forward_vector_angle` is deprecated and will "
        "be removed in a future release. Use "
        "`compute_perpendicular_vector` followed by "
        "`compute_vector_angle` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    forward_vector = compute_perpendicular_vector(
        data,
        left_keypoint,
        right_keypoint,
        camera_view=camera_view,
    )
    angle = compute_vector_angle(
        forward_vector,
        reference_vector=reference_vector,
        in_degrees=in_degrees,
    )
    angle.name = "forward_vector_angle"
    return angle


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
