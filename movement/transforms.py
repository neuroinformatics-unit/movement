"""Compute and apply spatial transforms."""

import itertools
from functools import partial
from typing import cast

import cv2
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from movement.utils.logging import log_to_attrs, logger
from movement.utils.vector import compute_signed_angle_2d
from movement.validators.arrays import validate_dims_coords


@log_to_attrs
def scale(
    data: xr.DataArray,
    factor: ArrayLike | float,
    space_unit: str | None = None,
) -> xr.DataArray:
    """Scale data by a given factor with an optional unit.

    Parameters
    ----------
    data
        The input data to be scaled.
    factor
        The scaling factor to apply to the data. If factor is a scalar (a
        single float), the data array is uniformly scaled by the same factor.
        If factor is an object that can be converted to a 1D numpy array (e.g.
        a list of floats), the length of the resulting array must match the
        length of data array's space dimension along which it will be
        broadcasted.
    space_unit
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


def poses_to_bboxes(
    position: xr.DataArray,
    padding: float = 0.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute bounding box centroid and shape from a poses position array.

    This function computes bounding boxes from pose estimation keypoints by
    finding the minimum and maximum coordinates across all keypoints for each
    individual at each time point. The resulting bounding box is represented
    by its centroid (center point) and shape (width and height).

    Parameters
    ----------
    position : xarray.DataArray
        A 2D poses position array with dimensions
        ``(time, space, keypoint, individual)``, where the ``space``
        coordinate contains exactly ``["x", "y"]``.
    padding : float, optional
        Number of pixels to add as padding around the bounding box in all
        directions. The padding increases both width and height by
        ``2 * padding``. Default is 0.0 (no padding).

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray]
        A tuple ``(position, shape)`` where:

        - ``position``: bounding box centroids with dimensions
          ``(time, space, individual)``.
        - ``shape``: bounding box width and height with dimensions
          ``(time, space, individual)``.

    Raises
    ------
    TypeError
        If ``position`` is not an :class:`xarray.DataArray` or if
        ``padding`` is not numeric.
    ValueError
        If the position array is missing required dimensions or coordinates,
        is not 2D, or ``padding`` is negative.

    Notes
    -----
    - Keypoints with NaN in any spatial coordinate are excluded from bounding
      box calculation. If all keypoints for an individual at a given time are
      NaN, the resulting centroid and shape are NaN.
    - The centroid is calculated as the midpoint of the bounding box:
      ``(min + max) / 2`` for each spatial dimension.
    - The shape is calculated as the span of coordinates plus padding:
      ``width = max_x - min_x + 2*padding`` and
      ``height = max_y - min_y + 2*padding``.
    - When there is only one valid keypoint, the bounding box will have
      zero width and/or height (before padding is applied).

    Examples
    --------
    Compute bounding boxes from a poses dataset with zero padding:

    >>> from movement.transforms import poses_to_bboxes
    >>> bbox_position, bbox_shape = poses_to_bboxes(poses_ds["position"])

    Compute bounding boxes from a poses dataset with 10 pixels of padding:

    >>> bbox_position, bbox_shape = poses_to_bboxes(
    ...     poses_ds["position"], padding=10
    ... )

    See Also
    --------
    movement.transforms.scale : Scale spatial coordinates

    """
    if not isinstance(position, xr.DataArray):
        raise TypeError(
            f"Expected an xarray DataArray, but got {type(position)}."
        )
    validate_dims_coords(
        position,
        {"time": [], "space": ["x", "y"], "keypoint": [], "individual": []},
        exact_coords=True,
    )
    if not isinstance(padding, int | float):
        raise TypeError(
            f"padding must be a number, got {type(padding).__name__}"
        )
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")

    # A keypoint is valid only if all spatial coordinates are present.
    valid_mask = ~position.isnull().any(dim="space")
    masked = position.where(valid_mask)

    pos_min = masked.min(dim="keypoint", skipna=True)
    pos_max = masked.max(dim="keypoint", skipna=True)

    centroid = (pos_min + pos_max) / 2
    shape = pos_max - pos_min + 2 * padding

    return centroid, shape


def compute_homography_transform(
    src_points: np.ndarray, dst_points: np.ndarray
) -> np.ndarray:
    """Compute a homography transformation matrix.

    Parameters
    ----------
    src_points
        An array of shape (N, 2) representing N source points
        in 2-dimensional space. N >= 4.
    dst_points
        An array of shape (N, 2) representing N destination points
        in 2-dimensional space. N >= 4.

    Returns
    -------
    numpy.ndarray
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
    perspective effects are present - e.g., when the camera is tilted,
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


def _rotation_matrix_2d_from_angle(
    angle: xr.DataArray, space_coord
) -> xr.DataArray:
    """Build 2D rotation matrices with dims (..., space_rot, space).

    `space_rot` indexes rows, `space` indexes columns
    """
    c = cast("xr.DataArray", np.cos(angle))
    s = cast("xr.DataArray", np.sin(angle))

    row_x = xr.concat([c, -s], dim="space").assign_coords(space=space_coord)
    row_y = xr.concat([s, c], dim="space").assign_coords(space=space_coord)

    return xr.concat([row_x, row_y], dim="space_rot").assign_coords(
        space_rot=space_coord
    )


class EgocentricAligner2d:
    """Align 2D pose tracks to an egocentric coordinate system.

    For each frame and individual, computes a centroid (the origin of the
    egocentric coordinate system) and a rotation angle (derived from the
    heading of ``keypoint_to_align`` relative to the new origin). That
    maps the input world coordinates onto a ego-centerically aligned
    coordinates such that its heading points along ``align_to_vector``.

    The fitted transform can be applied to position data via :meth:`align`,
    and inverted via :meth:`inverse_align`.

    Parameters
    ----------
    keypoint_to_align : str
        Keypoint whose position (relative to the centroid) defines the
        heading direction used to compute the rotation angle.
    keypoint_to_center : str or None
        Keypoint used to define the egocentric origin at each frame. If
        ``None``, the centroid is the mean position over all keypoints.
    align_to_vector : tuple of int, default (1, 0)
        The 2D vector that ``keypoint_to_align``'s heading is rotated to
        align with, expressed in (x, y) order. Defaults to the positive
        x-axis.

    Attributes
    ----------
    centroid_ : xarray.DataArray or None
        The fitted per-frame, per-individual centroid position. Set by
        :meth:`fit`; ``None`` before fitting.
    rotation_ : xarray.DataArray or None
        The fitted per-frame, per-individual 2D rotation matrices, with
        dims ``(..., space_rot, space)`` where ``space`` is the axis
        contracted over when applying the rotation. Set by :meth:`fit`;
        ``None`` before fitting.

    Notes
    -----
    This class borrows its ``fit``/``align``/``inverse_align`` structure from
    the scikit-learn transformer convention (``fit``/``transform``/
    ``inverse_transform``), even though ``fit`` is typically called on the
    same data subsequently passed to ``align`` rather than on a separate
    training set. Keeping the estimated transform as fitted state
    (:attr:`centroid_`, :attr:`rotation_`) rather than returning it directly
    means the same fitted transform can be reused later - e.g. applied to a
    different set of keypoints than the ones used to estimate it.

    """

    def __init__(
        self,
        keypoint_to_align: str,
        keypoint_to_center: str | None,
        align_to_vector: tuple[float, float] = (1, 0),
    ):
        """Create a new instance."""
        self.keypoint_to_align = keypoint_to_align
        self.keypoint_to_center = keypoint_to_center
        self.align_to_vector = np.asarray(align_to_vector)

        self.centroid_: xr.DataArray | None = None
        self.rotation_: xr.DataArray | None = (
            None  # dims: (..., space, space_rot)
        )

    def fit(self, position: xr.DataArray) -> "EgocentricAligner2d":
        """Compute the egocentric centroid and rotation matrix for each frame.

        Parameters
        ----------
        position : xarray.DataArray
            Position data with dims including ``space`` (with coordinates
            ``"x"`` and ``"y"``) and ``keypoint``.

        Returns
        -------
        EgocentricAligner2d
            self, with :attr:`centroid_` and :attr:`rotation_` populated.

        Raises
        ------
        ValueError
            If ``ds`` is missing the expected ``space`` coordinates or
            ``keypoint_to_align`` is not among ``ds``'s keypoints.

        """
        validate_dims_coords(position, {"space": ["x", "y"]})
        validate_dims_coords(position, {"keypoint": [self.keypoint_to_align]})

        if self.keypoint_to_center is None:
            self.centroid_ = position.mean(dim="keypoint")
        else:
            self.centroid_ = position.sel(keypoint=self.keypoint_to_center)

        centered = position - self.centroid_

        angles = compute_signed_angle_2d(
            centered.sel(keypoint=self.keypoint_to_align), self.align_to_vector
        )

        self.rotation_ = _rotation_matrix_2d_from_angle(
            angles, position.coords["space"].values
        )
        return self

    def _check_is_fitted(self):
        if self.rotation_ is None or self.centroid_ is None:
            raise RuntimeError(
                "Call `.fit(position)` before using the aligner."
            )

    def align(self, position: xr.DataArray) -> xr.DataArray:
        """Transform position data from world to egocentric coordinates.

        Parameters
        ----------
        position : xarray.DataArray
            Position data with a ``space`` dimension, sharing the
            coordinates system with the data used in :meth:`fit`.

        Returns
        -------
        xarray.DataArray
            ``position`` re-centred on the fitted centroid and rotated so
            that ``keypoint_to_align``'s heading points along
            ``align_to_vector``. Same dims/shape as the input.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.

        """
        self._check_is_fitted()
        assert self.centroid_ is not None
        assert self.rotation_ is not None

        # Center
        centered = position - self.centroid_

        # Apply rotation
        rotated = xr.dot(self.rotation_, centered, dim="space")

        # Clean-up and transpose to original order
        aligned = rotated.rename(space_rot="space").transpose(*position.dims)

        return aligned

    def inverse_align(self, position: xr.DataArray) -> xr.DataArray:
        """Transform position data from egocentric back to world coordinates.

        Applies the inverse of :meth:`align` - the transpose of the fitted
        rotation followed by adding back the fitted centroid.

        Parameters
        ----------
        position : xarray.DataArray
            Position data in the egocentric frame produced by :meth:`align`.

        Returns
        -------
        xarray.DataArray
            ``position`` mapped back into world coordinates. Same dims/
            shape as the input.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.

        """
        self._check_is_fitted()
        assert self.centroid_ is not None
        assert self.rotation_ is not None

        # Create inverse rotation by transposing the rotation matrices
        rot_T = self.rotation_.rename(space="space_tmp").rename(
            space_rot="space", space_tmp="space_rot"
        )

        # Apply inverse rotation
        rotated = xr.dot(rot_T, position, dim="space")

        # Clean-up and transpose to original order
        rotated = rotated.rename(space_rot="space").transpose(*position.dims)

        # Undo centering
        inverse_aligned = rotated + self.centroid_

        return inverse_aligned


class EgocentricAligner3d:
    """Align 3D pose tracks to an egocentric coordinate system.

    Unlike :class:`EgocentricAligner2d`, which derives a single heading
    angle from one keypoint, this aligner estimates a full 3D rotation by
    solving Wahba's problem (via :meth:`scipy.spatial.transform.Rotation.
    align_vectors`): given the centred positions of ``keypoints_to_align``
    at each frame, it finds the rotation that best maps them, in a
    weighted least-squares sense, onto the corresponding directions in
    ``align_to_vectors``. The fitted transform is applied to full position
    data via :meth:`align`, and inverted via :meth:`inverse_align`.

    Parameters
    ----------
    keypoints_to_align : list of str
        Keypoints whose (centered) positions define the orientation. At
        least two (non-collinear) keypoints are needed to fully determine
        the rotation; with only one, the rotation about that axis, defined
        by that keypoint, is left undetermined (scipy returns a valid but
        arbitrary solution along that axis).
    align_to_vectors : list of tuple of float
        Target (x, y, z) direction for each entry in ``keypoints_to_align``,
        in the same order, that its centred position is rotated towards.
        Must be the same length as ``keypoints_to_align``.
    alignment_weights : list of float, optional
        Per-keypoint weight in the least-squares rotation fit, in the same
        order as ``keypoints_to_align``. Must be the same length as
        ``keypoints_to_align`` if given. Defaults to equal weighting.
    keypoint_to_center : str or None, default None
        Keypoint used to define the egocentric origin at each frame. If
        ``None``, the centroid is the mean position over all keypoints.

    Attributes
    ----------
    centroid_ : xarray.DataArray or None
        The fitted per-frame, per-individual centroid position. Set by
        :meth:`fit`; ``None`` before fitting.
    rotation_ : xarray.DataArray or None
        Object-dtype array of :class:`scipy.spatial.transform.Rotation`,
        one per frame/individual. Set by :meth:`fit`; ``None`` before
        fitting.

    Raises
    ------
    ValueError
        If ``keypoints_to_align``, ``align_to_vectors``, and
        ``alignment_weights`` do not all have the same length.

    Notes
    -----
    This class borrows its ``fit``/``align``/``inverse_align`` structure from
    the scikit-learn transformer convention (``fit``/``transform``/
    ``inverse_transform``), see :class:`EgocentricAligner2d`

    Both ``align_to_vectors`` and the observed per-frame keypoint vectors
    (centred on the centroid) are normalized to unit length internally
    before the rotation fit, so only their *directions* - not their
    magnitudes - influence the estimated rotation. Relative importance
    between keypoints is controlled solely via ``alignment_weights``.

    """

    def __init__(
        self,
        keypoints_to_align: list[str],
        align_to_vectors: list[tuple[float, float, float]],
        alignment_weights: list[float] | None = None,
        keypoint_to_center: str | None = None,
    ):
        """Create a new instance and check alignment vector parameters."""
        n_alignment_vectors = len(keypoints_to_align)

        if n_alignment_vectors == 0:
            raise ValueError(
                "`keypoints_to_align` must have at least length one."
            )

        elif n_alignment_vectors == 1:
            logger.warning(
                "At least two (non-collinear) keypoints are needed to "
                "fully determine the rotation; with only one, the "
                "rotation about that axis, defined by that keypoint, "
                "is left undetermined"
            )

        if alignment_weights is None:
            alignment_weights = [1.0] * n_alignment_vectors

        if (
            len(align_to_vectors) != n_alignment_vectors
            or len(alignment_weights) != n_alignment_vectors
        ):
            raise ValueError(
                "`keypoints_to_align`, `align_to_vectors`, and "
                "`alignment_weights` must all have the same length."
            )

        self.keypoints_to_align = keypoints_to_align

        self.align_to_vectors = np.asarray(align_to_vectors)
        norms = np.linalg.norm(self.align_to_vectors, axis=1, keepdims=True)
        self.align_to_vectors = self.align_to_vectors / norms

        self.alignment_weights = np.asarray(alignment_weights)
        self.keypoint_to_center = keypoint_to_center

        self.centroid_: xr.DataArray | None = None
        self.rotation_: xr.DataArray | None = (
            None  # object-dtype array of Rotation
        )

    def fit(self, position: xr.DataArray) -> "EgocentricAligner3d":
        """Compute per-frame/individual centroid + rotation from da."""
        validate_dims_coords(position, {"space": ["x", "y", "z"]})
        validate_dims_coords(position, {"keypoint": self.keypoints_to_align})

        if self.keypoint_to_center is None:
            self.centroid_ = position.mean(dim="keypoint")
        else:
            self.centroid_ = position.sel(keypoint=self.keypoint_to_center)

        centered_positions = position - self.centroid_

        def estimate_rot_3d(v, ref_v):
            v_unit = v / np.linalg.norm(v, axis=0, keepdims=True)
            rot, _ = Rotation.align_vectors(
                ref_v, v_unit.T, weights=self.alignment_weights
            )
            return rot

        self.rotation_ = xr.apply_ufunc(
            partial(estimate_rot_3d, ref_v=self.align_to_vectors),
            centered_positions.sel(keypoint=self.keypoints_to_align),
            input_core_dims=[["space", "keypoint"]],
            output_core_dims=[[]],
            vectorize=True,
            output_dtypes=[Rotation],
        )
        return self

    def align(self, position: xr.DataArray) -> xr.DataArray:
        """World -> egocentric. Requires fit() first."""
        self._check_is_fitted()
        assert self.centroid_ is not None
        assert self.rotation_ is not None

        # Center
        centered_positions = position - self.centroid_

        # Apply rotation
        def apply_rot(v, rot):
            return rot.apply(v)

        aligned = xr.apply_ufunc(
            apply_rot,
            centered_positions,
            self.rotation_,
            input_core_dims=[["space"], []],
            output_core_dims=[["space"]],
            vectorize=True,
        )

        # Clean-up and transpose to original order
        aligned = aligned.transpose(*position.dims)

        return aligned

    def inverse_align(self, position: xr.DataArray) -> xr.DataArray:
        """Egocentric -> world. Requires fit() first."""
        self._check_is_fitted()
        assert self.centroid_ is not None
        assert self.rotation_ is not None

        # Apply inverse rotation
        def apply_inv_rot(v, rot):
            return rot.inv().apply(v)

        inverse_rotated = xr.apply_ufunc(
            apply_inv_rot,
            position,
            self.rotation_,
            input_core_dims=[["space"], []],
            output_core_dims=[["space"]],
            vectorize=True,
        )

        # Clean-up and transpose to original order
        inverse_rotated = inverse_rotated.transpose(*position.dims)

        # Undo centering
        inverse_aligned = inverse_rotated + self.centroid_

        return inverse_aligned

    def _check_is_fitted(self):
        if self.rotation_ is None or self.centroid_ is None:
            raise RuntimeError(
                "Call `.fit(position)` before using the aligner."
            )
