"""Compute kinematic variables like velocity and acceleration."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from movement.utils.logging import log_error
from movement.utils.vector import convert_to_unit
from movement.validators.arrays import validate_dims_coords


def compute_displacement(data: xr.DataArray) -> xr.DataArray:
    """Compute displacement array in cartesian coordinates.

    The displacement array is defined as the difference between the position
    array at time point ``t`` and the position array at time point ``t-1``.

    As a result, for a given individual and keypoint, the displacement vector
    at time point ``t``, is the vector pointing from the previous
    ``(t-1)`` to the current ``(t)`` position, in cartesian coordinates.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing displacement vectors in cartesian
        coordinates.

    Notes
    -----
    For the ``position`` array of a ``poses`` dataset, the ``displacement``
    array will hold the displacement vectors for every keypoint and every
    individual.

    For the ``position`` array of a ``bboxes`` dataset, the ``displacement``
    array will hold the displacement vectors for the centroid of every
    individual bounding box.

    For the ``shape`` array of a ``bboxes`` dataset, the
    ``displacement`` array will hold vectors with the change in width and
    height per bounding box, between consecutive time points.

    """
    validate_dims_coords(data, {"time": [], "space": []})
    result = data.diff(dim="time")
    result = result.reindex(data.coords, fill_value=0)
    return result


def compute_velocity(data: xr.DataArray) -> xr.DataArray:
    """Compute velocity array in cartesian coordinates.

    The velocity array is the first time-derivative of the position
    array. It is computed by applying the second-order accurate central
    differences method on the position array.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing velocity vectors in cartesian
        coordinates.

    Notes
    -----
    For the ``position`` array of a ``poses`` dataset, the ``velocity`` array
    will hold the velocity vectors for every keypoint and every individual.

    For the ``position`` array of a ``bboxes`` dataset, the ``velocity`` array
    will hold the velocity vectors for the centroid of every individual
    bounding box.

    See Also
    --------
    compute_time_derivative : The underlying function used.

    """
    # validate only presence of Cartesian space dimension
    # (presence of time dimension will be checked in compute_time_derivative)
    validate_dims_coords(data, {"space": []})
    return compute_time_derivative(data, order=1)


def compute_acceleration(data: xr.DataArray) -> xr.DataArray:
    """Compute acceleration array in cartesian coordinates.

    The acceleration array is the second time-derivative of the
    position array. It is computed by applying the second-order accurate
    central differences method on the velocity array.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing position information, with ``time``
        and ``space`` (in Cartesian coordinates) as required dimensions.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing acceleration vectors in cartesian
        coordinates.

    Notes
    -----
    For the ``position`` array of a ``poses`` dataset, the ``acceleration``
    array will hold the acceleration vectors for every keypoint and every
    individual.

    For the ``position`` array of a ``bboxes`` dataset, the ``acceleration``
    array will hold the acceleration vectors for the centroid of every
    individual bounding box.

    See Also
    --------
    compute_time_derivative : The underlying function used.

    """
    # validate only presence of Cartesian space dimension
    # (presence of time dimension will be checked in compute_time_derivative)
    validate_dims_coords(data, {"space": []})
    return compute_time_derivative(data, order=2)


def compute_time_derivative(data: xr.DataArray, order: int) -> xr.DataArray:
    """Compute the time-derivative of an array using numerical differentiation.

    This function uses :meth:`xarray.DataArray.differentiate`,
    which differentiates the array with the second-order
    accurate central differences method.

    Parameters
    ----------
    data : xarray.DataArray
        The input data containing ``time`` as a required dimension.
    order : int
        The order of the time-derivative. For an input containing position
        data, use 1 to compute velocity, and 2 to compute acceleration. Value
        must be a positive integer.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the time-derivative of the input data.

    See Also
    --------
    :meth:`xarray.DataArray.differentiate` : The underlying method used.

    """
    if not isinstance(order, int):
        raise log_error(
            TypeError, f"Order must be an integer, but got {type(order)}."
        )
    if order <= 0:
        raise log_error(ValueError, "Order must be a positive integer.")
    validate_dims_coords(data, {"time": []})
    result = data
    for _ in range(order):
        result = result.differentiate("time")
    return result


def compute_forward_vector(
    data: xr.DataArray,
    left_keypoint: str,
    right_keypoint: str,
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
    if camera_view == "top_down":
        upward_vector = np.array([0, 0, -1])
    else:
        upward_vector = np.array([0, 0, 1])

    upward_vector = xr.DataArray(
        np.tile(upward_vector.reshape(1, -1), [len(data.time), 1]),
        dims=["time", "space"],
    )

    # Compute forward direction as the cross product
    # (right-to-left) cross (forward) = up
    forward_vector = xr.cross(
        right_to_left_vector, upward_vector, dim="space"
    )[:, :, :-1]  # keep only the first 2 dimensions of the result

    # Return unit vector

    return convert_to_unit(forward_vector)


def compute_head_direction_vector(
    data: xr.DataArray,
    left_keypoint: str,
    right_keypoint: str,
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
) -> xr.DataArray:
    """Compute the 2D head direction vector given two keypoints on the head.

    This function is an alias for :func:`compute_forward_vector()\
    <movement.analysis.kinematics.compute_forward_vector>`. For more
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

    """
    return compute_forward_vector(
        data, left_keypoint, right_keypoint, camera_view=camera_view
    )


def compute_heading(
    data: xr.DataArray,
    left_keypoint: str,
    right_keypoint: str,
    reference_vector: npt.NDArray | list | tuple = (1, 0),
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
    in_radians=False,
) -> xr.DataArray:
    """Compute the 2D heading given two keypoints on the head.

    Heading is defined as the signed angle between the animal's forward
    vector (see :func:`compute_forward_direction()\
    <movement.analysis.kinematics.compute_forward_direction>`)
    and a reference vector. By default, the reference vector
    corresponds to the direction of the positive x-axis.

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two symmetrical keypoints located on the left and
        right sides of the body, respectively.
    left_keypoint : str
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint : str
        Name of the right keypoint, e.g., "right_ear"
    reference_vector : ndt.NDArray | list | tuple, optional
        The reference vector against which the ```forward_vector`` is
        compared to compute 2D heading. Must be a two-dimensional vector,
        in the form [x,y] - where reference_vector[0] corresponds to the
        x-coordinate and reference_vector[1] corresponds to the
        y-coordinate. If left unspecified, the vector [1, 0] is used by
        default.
    camera_view : Literal["top_down", "bottom_up"], optional
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.
    in_radians : bool, optional
        If true, the returned heading array is given in radians.
        If false, the array is given in degrees. False by default.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed heading
        timeseries, with dimensions matching the input data array,
        but without the ``keypoints`` and ``space`` dimensions.

    """
    # Convert reference vector to np.array if list or tuple
    if isinstance(reference_vector, (list | tuple)):
        reference_vector = np.array(reference_vector)

    # Validate that reference vector has correct dimensionality and type
    if reference_vector.shape != (2,):
        raise log_error(
            ValueError,
            f"Reference vector must be two-dimensional (with"
            f" shape `(2,)`), but got {reference_vector.shape}.",
        )
    if not (reference_vector.dtype == int) or (
        reference_vector.dtype == float
    ):
        raise log_error(
            ValueError,
            "Reference vector may only contain values of type ``int``"
            "or ``float``.",
        )

    # Compute forward vector and separate x and y components
    forward_vector = compute_forward_vector(
        data, left_keypoint, right_keypoint, camera_view=camera_view
    )
    forward_x = forward_vector.sel(space="x")
    forward_y = forward_vector.sel(space="y")

    # Normalize reference vector and separate x and y components
    reference_vector = reference_vector / np.linalg.norm(reference_vector)
    ref_x = reference_vector[0]
    ref_y = reference_vector[1]

    # Compute perp dot product to find signed angular difference between
    # forward vector and reference vector
    heading_array = np.arctan2(
        forward_y * ref_x - forward_x * ref_y,
        forward_x * ref_x + forward_y * ref_y,
    )

    # Convert to degrees
    if not in_radians:
        heading_array = np.rad2deg(heading_array)

    return heading_array


def compute_relative_heading(
    data: xr.DataArray,
    left_keypoint: str,
    right_keypoint: str,
    ROI: xr.DataArray | np.ndarray,
    camera_view: Literal["top_down", "bottom_up"] = "top_down",
    in_radians: bool = False,
) -> xr.DataArray:
    """Compute the 2D heading relative to an ROI.

    Relative heading is computed as the signed angle between
    the animal's forward vector (see :func:`compute_forward_direction()\
    <movement.analysis.kinematics.compute_forward_direction>`)
    and the vector pointing from the midpoint between the two provided
    left and right keypoints towards a region of interest (ROI).

    Parameters
    ----------
    data : xarray.DataArray
        The input data representing position. This must contain
        the two symmetrical keypoints located on the left and
        right sides of the body, respectively.
    left_keypoint : str
        Name of the left keypoint, e.g., "left_ear"
    right_keypoint : str
        Name of the right keypoint, e.g., "right_ear"
    ROI : xarray.DataArray | np.ndarray
        The position array for the region of interest against
        which heading will be computed. Position may be provided in the form of
        an ``xarray.DataArray`` (containing ``time``, ``space``, and optionally
        , ``keypoints`` dimensions) or a ``numpy.ndarray`` with the following
        axes: 0: time, 1: space (x, y), and (optionally) 2: keypoints. In both
        cases, if the input ROI contains multiple keypoints (e.g. the vertices
        of a bounding box), a centroid will be computed and the heading
        computed relative to this centroid. For ROIs provided as
        ``xarray.DataArray``'s, the time dimension must be equal in length to
        ``data.time``. For ROIs given as ``numpy.ndarray``'s, the time
        dimension must either have length 1 (e.g. a fixed coordinate for which
        to compute relative heading across a recording) or be equal in length
        to ``data.time``. For ``ndarray``'s with a single time point, take care
        to ensure the required axes are adhered to (e.g. ``np.array([[0,1]]))``
        is a valid ROI, while ``np.array([0,1])`` is not). Note also that the
        provided ROI position array may only contain one individual.
    camera_view : Literal["top_down", "bottom_up"], optional
        The camera viewing angle, used to determine the upwards
        direction of the animal. Can be either ``"top_down"`` (where the
        upwards direction is [0, 0, -1]), or ``"bottom_up"`` (where the
        upwards direction is [0, 0, 1]). If left unspecified, the camera
        view is assumed to be ``"top_down"``.
    in_radians : bool, optional
        If true, the returned heading array is given in radians.
        If false, the array is given in degrees. False by default.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray containing the computed relative heading
        timeseries, with dimensions matching the input data array,
        but without the ``keypoints`` and ``space`` dimensions.

    """
    # Validate ROI
    _validate_roi_for_relative_heading(ROI, data)

    # Drop individuals dim if present
    if "individuals" in data.dims:
        data = data.sel(individuals=data.individuals.values[0], drop=True)

    # Compute forward vector
    heading_vector = compute_forward_vector(
        data, left_keypoint, right_keypoint, camera_view=camera_view
    )
    forward_x = heading_vector.sel(space="x")
    forward_y = heading_vector.sel(space="y")

    # Get ROI coordinates
    if isinstance(ROI, xr.DataArray):
        # If ROI has a keypoints dimension, compute centroid over provided
        # points
        if "keypoints" in ROI.dims:
            ROI_coords = ROI.mean(dim="keypoints")
        else:
            ROI_coords = ROI
    else:
        # If ROI has a keypoints axis, compute centroid over provided points
        if len(ROI.shape) > 2:
            ROI = np.mean(ROI, axis=2)

        # If single timepoint, tile ROI array to match dimensions of ``data``
        if ROI.shape[0] == 1:
            ROI_coords = np.tile(ROI, [len(data.time), 1])
        else:
            ROI_coords = ROI

    # Compute reference vectors from Left-Right-Midpoint to ROI
    left_right_midpoint = data.sel(
        keypoints=[left_keypoint, right_keypoint]
    ).mean(dim="keypoints")

    reference_vectors = convert_to_unit(ROI_coords - left_right_midpoint)
    ref_x = reference_vectors.sel(space="x")
    ref_y = reference_vectors.sel(space="y")

    # Compute perp dot product to find signed angular difference between
    # forward vector and reference vector
    rel_heading_array = np.arctan2(
        forward_y * ref_x - forward_x * ref_y,
        forward_x * ref_x + forward_y * ref_y,
    )

    # Convert to degrees
    if not in_radians:
        rel_heading_array = np.rad2deg(rel_heading_array)

    return rel_heading_array


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


def _validate_roi_for_relative_heading(
    ROI: xr.DataArray | np.ndarray, data: xr.DataArray
):
    """Validate the ROI has the correct type and dimensions.

    Parameters
    ----------
    ROI : xarray.DataArray | numpy.ndarray
        The ROI position array to validate.
    data : xarray.DataArray
        The input data against which to validate the ROI.

    Returns
    -------
    TypeError
        If ROI is not an xarray.DataArray or a numpy.ndarray
    ValueError
        If ROI does not have the correct dimensions

    """
    if not isinstance(ROI, (xr.DataArray | np.ndarray)):
        raise log_error(
            TypeError,
            f"ROI must be an xarray.DataArray or a np.ndarray, but got "
            f"{type(data)}.",
        )
    if isinstance(ROI, xr.DataArray):
        validate_dims_coords(
            ROI,
            {
                "time": [],
                "space": [],
            },
        )
        if not len(ROI.time) == len(data.time):
            raise log_error(
                ValueError,
                "Input data and ROI must have matching time dimensions.",
            )
        if "individuals" in ROI.dims and len(ROI.individuals) > 1:
            raise log_error(
                ValueError, "ROI may not contain multiple individuals."
            )
    else:
        if not (
            ROI.shape[0] == 1 or ROI.shape[0] == len(data.time)
        ):  # Validate time dim
            raise log_error(
                ValueError,
                "Dimension ``0`` of the ``ROI`` argument must have length 1 or"
                " be equal in length to the ``time`` dimension of ``data``. \n"
                "\n If passing a single coordinate, make sure that ... (e.g. "
                "``np.array([[0,1]])``",
            )
        if not ROI.shape[1] == 2:  # Validate space dimension
            raise log_error(
                ValueError,
                "Dimension ``1`` of the ``ROI`` argument must correspond to "
                "coordinates in 2-D space, and may therefore only have size "
                f"``2``. Instead, got size ``{ROI.shape[1]}``.",
            )
        if len(ROI.shape) > 3:
            raise log_error(
                ValueError,
                "ROI may not have more than 3 dimensions (O: time, 1: space, "
                "2: keypoints).",
            )
