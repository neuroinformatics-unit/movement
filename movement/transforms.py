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
    factor : ArrayLike, float, or xarray.DataArray
        The scaling factor to apply to the data. If factor is a scalar (a
        single float), the data array is uniformly scaled by the same factor.
        If factor is a 1D array-like object, broadcasting follows
        xarray / NumPy default rules.
        If factor is an ``xarray.DataArray``, its dimensions are
        aligned by name with ``data`` and broadcasting is handled
        automatically by xarray.
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

    >>> factor = xr.DataArray([1.0, 2.0, 3.0], dims=("time",))
    >>> ds["position"] = scale(ds["position"], factor=factor)

    Using an ``xarray.DataArray`` for ``factor`` is recommended when the
    intended broadcasting dimension should be explicit.
    """
    if len(data.coords["space"]) == 2:
        validate_dims_coords(data, {"space": ["x", "y"]})
    else:
        validate_dims_coords(data, {"space": ["x", "y", "z"]})

    if not np.isscalar(factor):
        if not isinstance(factor, xr.DataArray):
            factor = np.asarray(factor)
            if factor.ndim > 1:
                raise ValueError(
                    "Factor must be a scalar or 1D array, "
                    f"got array with {factor.ndim} dimensions."
                )

    scaled_data = data * factor

    if space_unit is not None:
        scaled_data.attrs["space_unit"] = space_unit
    else:
        scaled_data.attrs.pop("space_unit", None)
    return scaled_data


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
