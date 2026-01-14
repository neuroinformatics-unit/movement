"""Compute and apply spatial transforms."""

import itertools

import cv2
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.utils.logging import log_to_attrs
from movement.validators.arrays import validate_dims_coords


@log_to_attrs
def scale(
    data: xr.DataArray,
    factor: ArrayLike | float = 1.0,
    space_unit: str | None = None,
) -> xr.DataArray:
    """Scale data by a given factor with an optional unit.

    Parameters
    ----------
    data : xarray.DataArray
        The input data to be scaled.
    factor : ArrayLike or float
        The scaling factor to apply to the data. If factor is a scalar (a
        single float), the data array is uniformly scaled by the same factor.
        If factor is an object that can be converted to a 1D numpy array (e.g.
        a list of floats), the length of the resulting array must match the
        length of data array's space dimension along which it will be
        broadcasted.
    space_unit : str or None
        The unit of the scaled data stored as a property in
        ``xarray.DataArray.attrs['space_unit']``. In case of the default
        (``None``) the ``space_unit`` attribute is dropped.

    Returns
    -------
    xarray.DataArray
        The scaled data array.

    Notes
    -----
    This function makes two changes to the resulting data array's attributes
    (:attr:`xarray.DataArray.attrs`) each time it is called:

    - It sets the ``space_unit`` attribute to the value of the parameter
      with the same name, or removes it if ``space_unit=None``.
    - It adds a new entry to the ``log`` attribute of the data array, which
      contains a record of the operations performed, including the
      parameters used, as well as the datetime of the function call.

    Examples
    --------
    Let's imagine a camera viewing a 2D plane from the top, with an
    estimated resolution of 10 pixels per cm. We can scale down
    position data by a factor of 1/10 to express it in cm units.

    >>> from movement.transforms import scale
    >>> ds["position"] = scale(ds["position"], factor=1 / 10, space_unit="cm")
    >>> print(ds["position"].space_unit)
    cm
    >>> print(ds["position"].log)
    [
        {
            "operation": "scale",
            "datetime": "2025-06-05 15:08:16.919947",
            "factor": "0.1",
            "space_unit": "'cm'"
        }
    ]

    Note that the attributes of the scaled data array now contain the assigned
    ``space_unit`` as well as a ``log`` entry with the arguments passed to
    the function.

    We can also scale the two spatial dimensions by different factors.

    >>> ds["position"] = scale(ds["position"], factor=[10, 20])

    The second scale operation restored the x axis to its original scale,
    and scaled up the y axis to twice its original size.
    The log will now contain two entries, but the ``space_unit`` attribute
    has been removed, as it was not provided in the second function call.

    >>> "space_unit" in ds["position"].attrs
    False

    """
    if len(data.coords["space"]) == 2:
        validate_dims_coords(data, {"space": ["x", "y"]})
    else:
        validate_dims_coords(data, {"space": ["x", "y", "z"]})

    if not np.isscalar(factor):
        factor = np.array(factor).squeeze()
        if factor.ndim != 1:
            raise ValueError(
                "Factor must be an object that can be converted to a 1D numpy"
                f" array, got {factor.ndim}D"
            )
        elif factor.shape != data.space.values.shape:
            raise ValueError(
                f"Factor shape {factor.shape} does not match the shape "
                f"of the space dimension {data.space.values.shape}"
            )
        else:
            factor_dims = [1] * data.ndim  # 1s array matching data dimensions
            factor_dims[data.get_axis_num("space")] = factor.shape[0]
            factor = factor.reshape(factor_dims)
    scaled_data = data * factor

    if space_unit is not None:
        scaled_data.attrs["space_unit"] = space_unit
    elif space_unit is None:
        scaled_data.attrs.pop("space_unit", None)
    return scaled_data


def _validate_poses_dataset(ds: xr.Dataset, padding_px: float) -> None:
    """Validate input dataset and parameters for poses_to_bboxes.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to validate.
    padding_px : float
        The padding value to validate.

    Raises
    ------
    TypeError
        If inputs are not of the expected types.
    ValueError
        If dataset structure or parameter values are invalid.
    """
    # Validate input type
    if not isinstance(ds, xr.Dataset):
        raise TypeError(
            f"Input must be an xarray.Dataset, got {type(ds).__name__}"
        )

    # Validate required data variable exists
    if "position" not in ds.data_vars:
        raise ValueError(
            "Input dataset must contain 'position' data variable. "
            f"Found data variables: {list(ds.data_vars)}"
        )

    # Validate dimensions and coordinates
    validate_dims_coords(ds.position, {"space": ["x", "y"]})

    # Check for 3D poses (not supported)
    if len(ds.coords["space"]) != 2:
        raise ValueError(
            "Input dataset must contain 2D poses only. "
            "Bounding boxes are inherently 2D and cannot be computed from 3D poses. "
            f"Found space dimension with coordinates: {list(ds.coords['space'].values)}"
        )

    # Validate padding parameter
    if not isinstance(padding_px, (int, float)):
        raise TypeError(
            f"padding_px must be a number, got {type(padding_px).__name__}"
        )
    if padding_px < 0:
        raise ValueError(f"padding_px must be non-negative, got {padding_px}")

    # Check required dimensions exist
    required_dims = ["time", "space", "keypoints", "individuals"]
    missing_dims = [dim for dim in required_dims if dim not in ds.position.dims]
    if missing_dims:
        raise ValueError(
            f"position data variable must have dimensions {required_dims}. "
            f"Missing: {missing_dims}"
        )


def _compute_bbox_for_keypoints(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    padding_px: float,
) -> tuple[float, float, float, float]:
    """Compute bounding box from keypoint coordinates.

    Parameters
    ----------
    x_coords : np.ndarray
        Array of x coordinates for keypoints.
    y_coords : np.ndarray
        Array of y coordinates for keypoints.
    padding_px : float
        Padding to add around the bounding box.

    Returns
    -------
    tuple[float, float, float, float]
        A tuple of (centroid_x, centroid_y, width, height).
        Returns NaN values if no valid keypoints exist.
    """
    # Filter out NaN values
    valid_x = x_coords[~np.isnan(x_coords)]
    valid_y = y_coords[~np.isnan(y_coords)]

    # Check if we have any valid keypoints
    if len(valid_x) == 0 or len(valid_y) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Compute bounding box
    x_min, x_max = valid_x.min(), valid_x.max()
    y_min, y_max = valid_y.min(), valid_y.max()

    # Centroid (center of bbox)
    centroid_x = (x_min + x_max) / 2
    centroid_y = (y_min + y_max) / 2

    # Shape (width, height) with padding
    width = x_max - x_min + 2 * padding_px
    height = y_max - y_min + 2 * padding_px

    return centroid_x, centroid_y, width, height


def _compute_bbox_confidence(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    kpt_confidences: np.ndarray,
) -> float:
    """Compute mean confidence for valid keypoints.

    Parameters
    ----------
    x_coords : np.ndarray
        Array of x coordinates for keypoints.
    y_coords : np.ndarray
        Array of y coordinates for keypoints.
    kpt_confidences : np.ndarray
        Array of confidence values for keypoints.

    Returns
    -------
    float
        Mean confidence of valid keypoints, or NaN if no valid keypoints exist.
    """
    # A keypoint is valid if BOTH x and y are not NaN
    valid_kpt_mask = ~np.isnan(x_coords) & ~np.isnan(y_coords)
    valid_confidences = kpt_confidences[valid_kpt_mask]

    # Mean of valid confidences (handles NaN in confidence array too)
    if len(valid_confidences) > 0:
        return np.nanmean(valid_confidences)
    return np.nan


def _get_confidence_data(ds: xr.Dataset) -> xr.DataArray:
    """Extract or create confidence data array.

    Parameters
    ----------
    ds : xarray.Dataset
        Input poses dataset.

    Returns
    -------
    xarray.DataArray
        Confidence data array with shape (time, keypoints, individuals).
    """
    if "confidence" in ds.data_vars:
        return ds.confidence
    # Create NaN confidence array if not present
    return xr.full_like(ds.position.isel(space=0), fill_value=np.nan)


def _create_bboxes_dataset(
    bbox_position: np.ndarray,
    bbox_shape: np.ndarray,
    bbox_confidence: np.ndarray,
    ds: xr.Dataset,
) -> xr.Dataset:
    """Create output bboxes dataset.

    Parameters
    ----------
    bbox_position : np.ndarray
        Array of bounding box positions (centroids).
    bbox_shape : np.ndarray
        Array of bounding box shapes (width, height).
    bbox_confidence : np.ndarray
        Array of bounding box confidences.
    ds : xarray.Dataset
        Original poses dataset (for coordinates and attributes).

    Returns
    -------
    xarray.Dataset
        The bboxes dataset.
    """
    dim_names = ("time", "space", "individuals")

    bboxes_ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                bbox_position,
                dims=dim_names,
                attrs=ds.position.attrs.copy(),  # Preserve attributes
            ),
            "shape": xr.DataArray(
                bbox_shape,
                dims=dim_names,
            ),
            "confidence": xr.DataArray(
                bbox_confidence,
                dims=(dim_names[0], dim_names[2]),  # (time, individuals)
            ),
        },
        coords={
            "time": ds.coords["time"],  # Preserve time coordinates
            "space": ["x", "y"],  # Always 2D for bboxes
            "individuals": ds.coords["individuals"],  # Preserve individual names
        },
        attrs=ds.attrs.copy(),  # Copy original attributes
    )

    # Update ds_type to indicate this is now a bboxes dataset
    bboxes_ds.attrs["ds_type"] = "bboxes"

    return bboxes_ds


@log_to_attrs
def poses_to_bboxes(
    ds: xr.Dataset,
    padding_px: float = 0.0,
) -> xr.Dataset:
    """Convert a poses dataset to a bounding boxes dataset.

    This function computes bounding boxes from pose estimation keypoints by
    finding the minimum and maximum coordinates across all keypoints for each
    individual at each time point. The resulting bounding box is represented
    by its centroid (center point) and shape (width and height).

    Parameters
    ----------
    ds : xarray.Dataset
        A ``movement`` poses dataset with dimensions
        (time, space, keypoints, individuals). The dataset must contain
        2D poses only (space dimension size = 2).
    padding_px : float, optional
        Number of pixels to add as padding around the bounding box in all
        directions. The padding increases both width and height by
        ``2 * padding_px``. Default is 0.0 (no padding).

    Returns
    -------
    xarray.Dataset
        A ``movement`` bboxes dataset with dimensions (time, space, individuals).
        The dataset contains:

        - ``position``: (n_frames, 2, n_individuals) array representing
          the centroid (x, y) of each bounding box.
        - ``shape``: (n_frames, 2, n_individuals) array representing
          the width and height of each bounding box.
        - ``confidence``: (n_frames, n_individuals) array with the mean
          confidence across all keypoints for each individual.

    Raises
    ------
    ValueError
        If the input dataset does not have the required 'position' data variable.
    ValueError
        If the input dataset contains 3D poses (space dimension size = 3).
        Only 2D poses are supported as bounding boxes are inherently 2D.
    ValueError
        If padding_px is negative.

    Notes
    -----
    - Keypoints with NaN coordinates are excluded from bounding box calculation.
      If all keypoints for an individual at a given time are NaN, the resulting
      bounding box position, shape, and confidence will all be NaN.
    - The confidence value for each bounding box is computed as the mean of
      the confidence values of all valid (non-NaN) keypoints for that
      individual at that time point.
    - The bounding box centroid is calculated as the midpoint between the
      minimum and maximum coordinates: ``(min + max) / 2``.
    - The bounding box shape is calculated as the span of coordinates plus
      padding: ``width = max_x - min_x + 2*padding_px`` and
      ``height = max_y - min_y + 2*padding_px``.
    - When there is only one valid keypoint, the bounding box will have
      zero width and/or height (before padding is applied).
    - The function preserves dataset attributes (time_unit, fps, source_software)
      from the input poses dataset, but updates the ``ds_type`` to "bboxes".
    - This function makes changes to the resulting dataset's attributes
      (:attr:`xarray.Dataset.attrs`):

        - It sets the ``ds_type`` attribute to "bboxes".
        - It adds a new entry to the ``log`` attribute, which contains
          a record of the operation performed, including parameters used
          and the datetime of the function call.

    Examples
    --------
    Convert a poses dataset to bounding boxes:

    >>> from movement.transforms import poses_to_bboxes
    >>> bboxes_ds = poses_to_bboxes(poses_ds)

    Add 10 pixels of padding around each bounding box:

    >>> bboxes_ds = poses_to_bboxes(poses_ds, padding_px=10)

    The resulting bounding boxes can be accessed via:

    >>> bboxes_ds.position  # centroids
    >>> bboxes_ds.shape     # widths and heights
    >>> bboxes_ds.confidence  # mean keypoint confidences

    See Also
    --------
    movement.filtering.filter_by_confidence : Filter data by confidence threshold
    movement.transforms.scale : Scale spatial coordinates

    """
    # Validate inputs
    _validate_poses_dataset(ds, padding_px)

    # Extract data
    position = ds.position  # shape: (time, space, keypoints, individuals)
    confidence_kpts = _get_confidence_data(ds)

    # Get dimensions
    n_frames = len(ds.coords["time"])
    n_individuals = len(ds.coords["individuals"])

    # Initialize output arrays with NaN
    bbox_position = np.full((n_frames, 2, n_individuals), np.nan)
    bbox_shape = np.full((n_frames, 2, n_individuals), np.nan)
    bbox_confidence = np.full((n_frames, n_individuals), np.nan)

    # Compute bounding boxes for each frame and individual
    for t_idx in range(n_frames):
        for ind_idx in range(n_individuals):
            # Extract keypoint positions for this individual at this time
            keypoints = position.isel(time=t_idx, individuals=ind_idx).values

            # Extract x and y coordinates
            x_coords = keypoints[0, :]  # All x values across keypoints
            y_coords = keypoints[1, :]  # All y values across keypoints

            # Compute bounding box
            centroid_x, centroid_y, width, height = _compute_bbox_for_keypoints(
                x_coords, y_coords, padding_px
            )

            # Store position and shape results
            bbox_position[t_idx, 0, ind_idx] = centroid_x
            bbox_position[t_idx, 1, ind_idx] = centroid_y
            bbox_shape[t_idx, 0, ind_idx] = width
            bbox_shape[t_idx, 1, ind_idx] = height

            # Compute and store confidence
            kpt_conf = confidence_kpts.isel(time=t_idx, individuals=ind_idx).values
            bbox_confidence[t_idx, ind_idx] = _compute_bbox_confidence(
                x_coords, y_coords, kpt_conf
            )

    # Create and return output dataset
    return _create_bboxes_dataset(
        bbox_position, bbox_shape, bbox_confidence, ds
    )


def compute_homography_transform(
    src_points: np.ndarray, dst_points: np.ndarray
) -> np.ndarray:
    """Compute a homography transformation matrix.

    Parameters
    ----------
    src_points : np.ndarray
        An array of shape (N, 2) representing N source points
        in 2-dimensional space. N >= 4.
    dst_points : np.ndarray
        An array of shape (N, 2) representing N destination points
        in 2-dimensional space. N >= 4.

    Returns
    -------
    np.ndarray
        A (3, 3) transformation matrix that aligns the
        source points to the destination points.

    Raises
    ------
    ValueError
        If the number of source points does not match
            the number of destination points,
        or if there are insufficient points to
            compute the transformation,
        or if the points are not 2-dimensional,
        or if the points are degenerate or collinear,
            making it impossible to compute a valid homography.

    Notes
    -----
    This function estimates a 3x3 homography matrix using corresponding 2D
    point pairs from two images or planes. A homography describes a
    projective transformation suitable for **planar scenes** where
    perspective effects are present — e.g., when the camera is tilted,
    moved closer, or rotated relative to the plane.

    The transformation preserves straight lines but not
    necessarily parallelism or distances, making it ideal for:

    - Image rectification
    - Perspective warping
    - Planar object tracking

    Important considerations:

    - At least **4 non-collinear, non-degenerate points** are
      required to compute a valid homography transformation.
    - The function internally filters invalid point pairs.
    - The computed homography is most accurate for
      **planar scenes with perspective distortion**,
      where the transformation can be modeled
      as a projective mapping.

    Examples
    --------
    >>> src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    >>> dst = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    >>> H = compute_homography_transform(src, dst)
    >>> print(H.shape)
    (3, 3)

    """
    src_points = np.asarray(src_points, dtype=np.float32)
    dst_points = np.asarray(dst_points, dtype=np.float32)

    _validate_points_shape(src_points, dst_points)

    src_points, dst_points = _filter_invalid_points(src_points, dst_points)
    num_points = src_points.shape[0]

    if num_points < 4:
        raise ValueError(
            "Insufficient points to compute the homography transformation."
        )

    transform_matrix, _ = cv2.findHomography(
        src_points, dst_points, method=cv2.RANSAC
    )

    return transform_matrix


def _validate_points_shape(src_points: np.ndarray, dst_points: np.ndarray):
    """Validate that source and destination point arrays.

    The arrays should have matching 2D shapes.
    """
    if len(src_points.shape) != 2 or len(dst_points.shape) != 2:
        raise ValueError("Points must be 2-dimensional arrays.")

    if src_points.shape != dst_points.shape:
        raise ValueError(
            "Source and destination points must have the same shape."
        )

    dim = src_points.shape[1]
    if dim != 2:
        raise ValueError("Points must be 2-dimensional.")


def _filter_invalid_points(src_pts: np.ndarray, dst_pts: np.ndarray):
    """Remove invalid points.

    Invalid points are duplicate, degenerate, or
    collinear point pairs from the input sets.
    """
    keep_idx: list[int] = []
    obtained_min_non_collinear_set = False
    eps = 1e-6

    for i in range(len(src_pts)):
        # skip duplicates
        if any(
            np.linalg.norm(src_pts[i] - src_pts[j]) < eps for j in keep_idx
        ):
            continue

        subset = np.vstack([src_pts[j] for j in keep_idx] + [src_pts[i]])

        if subset.shape[0] < 3:
            keep_idx.append(i)
            continue
        elif subset.shape[0] == 3 and _is_collinear_set(subset, eps):
            continue

        # If we have at least 3 old points, check that
        # new point is not collinear with any other two
        if not obtained_min_non_collinear_set and subset.shape[0] > 3:
            all_noncollinear_triples = all(
                not _is_collinear_three(
                    src_pts[a], src_pts[b], src_pts[i], eps
                )
                for a, b in itertools.combinations(keep_idx, 2)
            )
            if all_noncollinear_triples:
                obtained_min_non_collinear_set = True
            else:
                continue
        keep_idx.append(i)

    return src_pts[keep_idx], dst_pts[keep_idx]


def _is_collinear_three(a, b, c, eps):
    """Check if three 2D points are collinear via the cross-product method."""
    return (
        abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        <= eps
    )


def _is_collinear_set(points: np.ndarray, eps):
    """Check if a set of 2D points is collinear.

    Uses singular value decomposition (SVD) to determine
    whether all points lie on a single straight line.
    """
    pts = np.array(points)
    pts -= pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts)
    rank = np.sum(s > eps)
    return rank < 2
