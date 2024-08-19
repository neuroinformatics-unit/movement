"""Extract variables useful for the study of spatial navigation."""

import numpy as np
import xarray as xr

from movement.utils.logging import log_error


def compute_head_direction_vector(
    data: xr.DataArray, left_keypoint: str, right_keypoint: str
):
    """Compute the head direction vector given two keypoints on the head.

    The head direction vector is computed as a vector perpendicular to the
    line connecting two keypoints on either side of the head, pointing
    forwards (in a rostral direction).

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

    Returns
    -------
    xarray.DataArray
        An xarray DataArray representing the head direction vector,
        with dimensions matching the input data array, but without the
        ``keypoints`` dimension.

    """
    # Validate input dataset
    if not isinstance(data, xr.DataArray):
        raise log_error(
            TypeError,
            f"Input data must be an xarray.DataArray, but got {type(data)}.",
        )
    if not all(coord in data.dims for coord in ["time", "keypoints", "space"]):
        raise log_error(
            AttributeError,
            "Input data must contain 'time', 'space', and 'keypoints' as "
            "dimensions.",
        )
    if not all(
        keypoint in data.keypoints
        for keypoint in [left_keypoint, right_keypoint]
    ):
        raise log_error(
            AttributeError,
            "The selected keypoints could not be found in the input dataset",
        )

    # Select the right and left keypoints
    head_left = data.sel(keypoints=left_keypoint, drop=True)
    head_right = data.sel(keypoints=right_keypoint, drop=True)

    # Initialize a vector from right to left ear, and another vector
    # perpendicular to the X-Y plane
    right_to_left_vector = head_left - head_right
    perpendicular_vector = np.array([0, 0, -1])

    # Compute cross product
    head_vector = head_right.copy()
    head_vector.values = np.cross(right_to_left_vector, perpendicular_vector)[
        :, :, :-1
    ]

    return head_vector
