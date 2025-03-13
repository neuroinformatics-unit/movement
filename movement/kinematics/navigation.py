"""Compute variables useful for spatial navigation analysis."""

from collections.abc import Hashable
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.kinematics.motion import compute_displacement
from movement.kinematics.utils import (
    _compute_scaled_path_length,
    _validate_type_data_array,
    _warn_about_nan_proportion,
)
from movement.utils.logging import log_error
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
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the forward vector, with
        dimensions matching the input data array, but without the
        ``keypoints`` dimension.

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
            "keypoints": [left_keypoint, right_keypoint],
            "space": [],
        },
    )
    if len(data.space) != 2:
        raise log_error(
            ValueError,
            "Input data must have exactly 2 spatial dimensions, but "
            f"currently has {len(data.space)}.",
        )
    # Validate input keypoints
    if left_keypoint == right_keypoint:
        raise log_error(
            ValueError, "The left and right keypoints may not be identical."
        )
    # Define right-to-left vector
    right_to_left_vector = data.sel(
        keypoints=left_keypoint, drop=True
    ) - data.sel(keypoints=right_keypoint, drop=True)
    # Define upward vector
    # default: negative z direction in the image coordinate system
    upward_vector = (
        np.array([0, 0, -1])
        if camera_view == "top_down"
        else np.array([0, 0, 1])
    )
    upward_vector = xr.DataArray(
        np.tile(upward_vector.reshape(1, -1), [len(data.time), 1]),
        dims=["time", "space"],
        coords={
            "space": ["x", "y", "z"],
        },
    )
    # Compute forward direction as the cross product
    # (right-to-left) cross (forward) = up
    forward_vector = xr.cross(
        right_to_left_vector, upward_vector, dim="space"
    ).drop_sel(
        space="z"
    )  # keep only the first 2 spatial dimensions of the result
    # Return unit vector
    return convert_to_unit(forward_vector)


def compute_head_direction_vector(
    data: xr.DataArray,
    left_keypoint: str,
    right_keypoint: str,
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
):
    """Compute the 2D head direction vector given two keypoints on the head.

    This function is an alias for :func:`compute_forward_vector`. For more
    detailed information on how the head direction vector is computed,
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
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the head direction vector, with
        dimensions matching the input data array, but without the
        ``keypoints`` dimension.

    See Also
    --------
    compute_forward_vector :
        The function used to compute the head direction vector.

    """
    return compute_forward_vector(
        data, left_keypoint, right_keypoint, camera_view=camera_view
    )


def compute_forward_vector_angle(
    data: xr.DataArray,
    left_keypoint: Hashable,
    right_keypoint: Hashable,
    reference_vector: xr.DataArray | ArrayLike = (1, 0),
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
    in_degrees: bool = False,
) -> xr.DataArray:
    r"""Compute the signed angle between a forward and reference vector.

    Forward vector angle is the :func:`signed angle\
    <movement.utils.vector.compute_signed_angle_2d>`
    between the reference vector and the animal's :func:`forward vector\
    <compute_forward_vector>`.
    The returned angles are in radians, spanning the range :math:`(-\pi, \pi]`,
    unless ``in_degrees`` is set to ``True``.

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two symmetrical keypoints located on the left and
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
        view is assumed to be ``"top_down"``
    in_degrees : bool
        If ``True``, the returned heading array is given in degrees.
        Otherwise, the array is given in radians. Default ``False``.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed forward vector angles,
        with dimensions matching the input data array,
        but without the ``keypoints`` and ``space`` dimensions.

    See Also
    --------
    movement.utils.vector.compute_signed_angle_2d :
        The underlying function used to compute the signed angle between two
        2D vectors. See this function for a definition of the signed
        angle between two vectors.
    compute_forward_vector :
        The function used to compute the forward vector.

    """
    # Convert reference vector to np.array if not already a valid array
    if not isinstance(reference_vector, np.ndarray | xr.DataArray):
        reference_vector = np.array(reference_vector)

    # Compute forward vector
    forward_vector = compute_forward_vector(
        data, left_keypoint, right_keypoint, camera_view=camera_view
    )

    # Compute signed angle between forward vector and reference vector
    heading_array = compute_signed_angle_2d(forward_vector, reference_vector)

    # Convert to degrees
    if in_degrees:
        heading_array = np.rad2deg(heading_array)

    return heading_array


def compute_path_length(
    data: xr.DataArray,
    start: float | None = None,
    stop: float | None = None,
    nan_policy: Literal["ffill", "scale"] = "ffill",
    nan_warn_threshold: float = 0.2,
) -> xr.DataArray:
    """Compute the length of a path travelled between two time points.

    The path length is defined as the sum of the norms (magnitudes) of the
    displacement vectors between two time points ``start`` and ``stop``,
    which should be provided in the time units of the data array.
    If not specified, the minimum and maximum time coordinates of the data
    array are used as start and stop times, respectively.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.
    start : float, optional
        The start time of the path. If None (default),
        the minimum time coordinate in the data is used.
    stop : float, optional
        The end time of the path. If None (default),
        the maximum time coordinate in the data is used.
    nan_policy : Literal["ffill", "scale"], optional
        Policy to handle NaN (missing) values. Can be one of the ``"ffill"``
        or ``"scale"``. Defaults to ``"ffill"`` (forward fill).
        See Notes for more details on the two policies.
    nan_warn_threshold : float, optional
        If more than this proportion of values are missing in any point track,
        a warning will be emitted. Defaults to 0.2 (20%).

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed path length,
        with dimensions matching those of the input data,
        except ``time`` and ``space`` are removed.

    Notes
    -----
    Choosing ``nan_policy="ffill"`` will use :meth:`xarray.DataArray.ffill`
    to forward-fill missing segments (NaN values) across time.
    This equates to assuming that a track remains stationary for
    the duration of the missing segment and then instantaneously moves to
    the next valid position, following a straight line. This approach tends
    to underestimate the path length, and the error increases with the number
    of missing values.

    Choosing ``nan_policy="scale"`` will adjust the path length based on the
    the proportion of valid segments per point track. For example, if only
    80% of segments are present, the path length will be computed based on
    these and the result will be divided by 0.8. This approach assumes
    that motion dynamics are similar across observed and missing time
    segments, which may not accurately reflect actual conditions.

    """
    validate_dims_coords(data, {"time": [], "space": []})
    data = data.sel(time=slice(start, stop))
    # Check that the data is not empty or too short
    n_time = data.sizes["time"]
    if n_time < 2:
        raise log_error(
            ValueError,
            f"At least 2 time points are required to compute path length, "
            f"but {n_time} were found. Double-check the start and stop times.",
        )

    _warn_about_nan_proportion(data, nan_warn_threshold)

    if nan_policy == "ffill":
        return compute_norm(
            compute_displacement(data.ffill(dim="time")).isel(
                time=slice(1, None)
            )  # skip first displacement (always 0)
        ).sum(dim="time", min_count=1)  # return NaN if no valid segment
    elif nan_policy == "scale":
        return _compute_scaled_path_length(data)
    else:
        raise log_error(
            ValueError,
            f"Invalid value for nan_policy: {nan_policy}. "
            "Must be one of 'ffill' or 'scale'.",
        )
