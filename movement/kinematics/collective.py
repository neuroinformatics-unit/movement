# collective.py
"""Compute collective behavior metrics for multi-individual tracking data."""

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from movement.utils.logging import logger
from movement.utils.vector import (
    compute_norm,
    compute_signed_angle_2d,
    convert_to_unit,
)
from movement.validators.arrays import validate_dims_coords

_ANGLE_EPS = 1e-12


@dataclass
class _ValidateAPConfig:
    """Configuration for the _validate_ap function.

    Parameters
    ----------
    min_valid_frac : float, default=0.6
        Minimum fraction of keypoints that must be present for a frame
        to qualify as tier-1 valid.
    window_len : int, default=50
        Number of speed samples per sliding window.
    stride : int, default=5
        Step size between consecutive sliding window start positions.
    pct_thresh : float, default=85.0
        Percentile threshold applied to valid-window median speeds for
        high-motion classification.
    min_run_len : int, default=1
        Minimum number of consecutive qualifying windows required to
        form a valid run.
    postural_var_ratio_thresh : float, default=2.0
        Between-segment to within-segment RMSD variance ratio above which
        postural clustering is triggered.
    max_clusters : int, default=4
        Upper bound on the number of clusters to evaluate during k-medoids.
    confidence_floor : float, default=0.1
        Vote margin below which the anterior inference is flagged as
        unreliable.
    lateral_thresh : float, default=0.4
        Normalized lateral offset ceiling for the Step 1 lateral alignment
        filter.
    edge_thresh : float, default=0.3
        Normalized midpoint distance floor for the Step 3 distal/proximal
        classification.

    """

    min_valid_frac: float = 0.6
    window_len: int = 50
    stride: int = 5
    pct_thresh: float = 85.0
    min_run_len: int = 1
    postural_var_ratio_thresh: float = 2.0
    max_clusters: int = 4
    confidence_floor: float = 0.1
    lateral_thresh: float = 0.4
    edge_thresh: float = 0.3

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate fraction parameters (must be in [0, 1])
        for name in (
            "min_valid_frac",
            "confidence_floor",
            "lateral_thresh",
            "edge_thresh",
        ):
            value = getattr(self, name)
            if not (0 <= value <= 1):
                raise ValueError(
                    f"{name} must be between 0 and 1, got {value}"
                )

        # Validate positive integer parameters
        for name in ("window_len", "stride", "min_run_len", "max_clusters"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer, got {value}"
                )

        # Validate pct_thresh (must be in [0, 100])
        if not (0 <= self.pct_thresh <= 100):
            raise ValueError(
                f"pct_thresh must be between 0 and 100, got {self.pct_thresh}"
            )

        # Validate postural_var_ratio_thresh (must be positive)
        if self.postural_var_ratio_thresh <= 0:
            raise ValueError(
                f"postural_var_ratio_thresh must be positive, "
                f"got {self.postural_var_ratio_thresh}"
            )


@dataclass
class _FrameSelection:
    """Selected frames from high-motion segmentation and tier-2 filtering.

    Bundles the frame indices, segment assignments, and related arrays
    produced by the segmentation pipeline for downstream consumption
    (skeleton construction, postural clustering, velocity recomputation).

    Attributes
    ----------
    frames : np.ndarray
        Array of selected frame indices (tier-2 valid, within segments).
    seg_ids : np.ndarray
        Segment ID (0-indexed) for each selected frame.
    segments : np.ndarray
        Array of shape (n_segments, 2) with [frame_start, frame_end].
    bbox_centroids : np.ndarray
        Array of shape (n_frames, 2) with bounding-box centroids.
    count : int
        Number of selected frames.

    """

    frames: np.ndarray
    seg_ids: np.ndarray
    segments: np.ndarray
    bbox_centroids: np.ndarray
    count: int


@dataclass
class _APNodePairReport:
    """Report from the AP node-pair evaluation pipeline.

    This dataclass holds all results from the 3-step filter cascade
    used to evaluate a candidate anterior-posterior keypoint pair.

    Attributes
    ----------
    success : bool
        Whether the evaluation pipeline completed successfully.
    failure_step : str
        Name of the step at which evaluation failed, if any.
    failure_reason : str
        Reason for failure, if any.
    scenario : int
        Scenario number (1-13) from the mutually exclusive outcomes.
    outcome : str
        Either "accept" or "warn".
    warning_message : str
        Warning message, if applicable.
    sorted_candidate_nodes : np.ndarray
        Indices of candidate nodes after Step 1 filtering, sorted by
        ascending normalized lateral offset.
    valid_pairs : np.ndarray
        Array of shape (n_pairs, 2) containing valid node pairs after
        Step 2 filtering.
    valid_pairs_internode_dist : np.ndarray
        Internode separation (AP distance) for each valid pair.
    input_pair_in_candidates : bool
        Whether the input pair survived Step 1 filtering.
    input_pair_opposite_sides : bool
        Whether the input pair lies on opposite sides of the midpoint.
    input_pair_separation_abs : float
        Absolute AP separation of the input pair.
    input_pair_is_distal : bool
        Whether the input pair is classified as distal in Step 3.
    input_pair_rank : int
        Rank of the input pair by internode separation (1 = largest).
    input_pair_order_matches_inference : bool
        Whether from_node has a lower AP coordinate than to_node
        (i.e. from_node is more posterior). True means the input pair
        ordering is consistent with the inferred AP axis.
    pc1_coords : np.ndarray
        PC1 coordinates for each keypoint.
    ap_coords : np.ndarray
        AP (anterior-posterior) coordinates for each keypoint.
    lateral_offsets : np.ndarray
        Unsigned lateral offset from body axis for each keypoint.
    lateral_offsets_norm : np.ndarray
        Normalized lateral offsets (0 = nearest to axis, 1 = farthest).
    lateral_offset_min : float
        Minimum lateral offset among valid keypoints.
    lateral_offset_max : float
        Maximum lateral offset among valid keypoints.
    midpoint_pc1 : float
        AP reference midpoint (average of min and max PC1 projections).
    pc1_min : float
        Minimum PC1 projection among valid keypoints.
    pc1_max : float
        Maximum PC1 projection among valid keypoints.
    midline_dist_norm : np.ndarray
        Normalized distance from midpoint for each keypoint.
    midline_dist_max : float
        Maximum absolute distance from midpoint.
    distal_pairs : np.ndarray
        Array of distal pairs (both nodes at or above edge_thresh).
    proximal_pairs : np.ndarray
        Array of proximal pairs (at least one node below edge_thresh).
    max_separation_distal_nodes : np.ndarray
        Node indices of the maximum-separation distal pair, ordered
        so that element 0 is posterior (lower AP coord) and element 1
        is anterior (higher AP coord).
    max_separation_distal : float
        Internode separation of the max-separation distal pair.
    max_separation_nodes : np.ndarray
        Node indices of the overall maximum-separation pair, ordered
        so that element 0 is posterior (lower AP coord) and element 1
        is anterior (higher AP coord).
    max_separation : float
        Internode separation of the overall max-separation pair.

    """

    success: bool = False
    failure_step: str = ""
    failure_reason: str = ""
    scenario: int = 0
    outcome: str = ""
    warning_message: str = ""

    sorted_candidate_nodes: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    valid_pairs: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=int)
    )
    valid_pairs_internode_dist: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    input_pair_in_candidates: bool = False
    input_pair_opposite_sides: bool = False
    input_pair_separation_abs: float = np.nan
    input_pair_is_distal: bool = False
    input_pair_rank: int = 0
    input_pair_order_matches_inference: bool = False

    pc1_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    ap_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    lateral_offsets: np.ndarray = field(default_factory=lambda: np.array([]))
    lateral_offsets_norm: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    lateral_offset_min: float = np.nan
    lateral_offset_max: float = np.nan
    midpoint_pc1: float = np.nan
    pc1_min: float = np.nan
    pc1_max: float = np.nan
    midline_dist_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    midline_dist_max: float = np.nan

    distal_pairs: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=int)
    )
    proximal_pairs: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=int)
    )
    max_separation_distal_nodes: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    max_separation_distal: float = np.nan
    max_separation_nodes: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    max_separation: float = np.nan


def compute_polarization(
    data: xr.DataArray,
    body_axis_keypoints: tuple[Hashable, Hashable] | None = None,
    displacement_frames: int = 1,
    return_angle: bool = False,
    in_degrees: bool = False,
    validate_ap: bool = True,
    ap_validation_config: dict[str, Any] | None = None,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    r"""Compute polarization (group alignment) of individuals.

    Polarization measures how aligned individuals' direction vectors are,
    supporting two modes: **orientation polarization** (body-axis mode) for
    body orientation alignment, and **heading polarization** (displacement
    mode) for movement direction alignment. A value of 1 indicates perfect
    alignment, while a value near 0 indicates weak or canceling alignment.

    The polarization is computed as

    .. math::

        \Phi = \frac{1}{N} \left\| \sum_{i=1}^{N} \hat{u}_i \right\|

    where :math:`\hat{u}_i` is the unit direction vector for individual
    :math:`i`, and :math:`N` is the number of valid individuals at that time.

    Parameters
    ----------
    data : xarray.DataArray
        Position data. Must contain ``time``, ``space``, and ``individuals`` as
        dimensions. If ``body_axis_keypoints`` is provided, the array must also
        contain a ``keypoints`` dimension. For displacement-based heading,
        pre-select a keypoint (e.g., ``data.sel(keypoints="thorax")``) or the
        first keypoint (index 0) will be used.

        Spatial coordinates must include ``"x"`` and ``"y"``. If additional
        spatial coordinates are present (e.g., ``"z"``), they are ignored;
        polarization is computed in the x/y plane.
    body_axis_keypoints : tuple[Hashable, Hashable], optional
        Pair of keypoint names ``(origin, target)`` used to compute heading as
        the vector from origin to target. If omitted, heading is inferred from
        displacement over ``displacement_frames``.
    displacement_frames : int, default=1
        Number of frames used to compute displacement when
        ``body_axis_keypoints`` is not provided. Must be a positive integer.
        This parameter is ignored when ``body_axis_keypoints`` is provided.
    return_angle : bool, default=False
        If True, also return the mean angle. Returns the mean body
        orientation angle when using ``body_axis_keypoints``, or the mean
        movement direction angle when using displacement-based polarization.
    in_degrees : bool, default=False
        If True, the mean angle is returned in degrees. Otherwise, the
        angle is returned in radians. Only relevant when
        ``return_angle=True``.
    validate_ap : bool, default=True
        If True, run anterior-posterior axis validation when
        ``body_axis_keypoints`` is provided. Validation is skipped for
        displacement-based polarization.
    ap_validation_config : dict, optional
        Configuration overrides for anterior-posterior axis validation.
        Passed to ``_ValidateAPConfig`` when validation is enabled.

    Returns
    -------
    xarray.DataArray or tuple[xarray.DataArray, xarray.DataArray]
        If ``return_angle`` is False, returns a DataArray named
        ``"polarization"`` with dimension ``("time",)``.

        If ``return_angle`` is True, returns
        ``(polarization, mean_angle)`` where ``mean_angle`` is a DataArray
        named ``"mean_angle"`` with dimension ``("time",)``.

    Notes
    -----
    Missing data are excluded per individual, per frame.

    Zero-length headings are treated as invalid and excluded from the
    calculation.

    The mean angle is defined from the summed unit-heading vector projected
    onto the x/y plane. When using ``body_axis_keypoints``, this represents
    the mean body orientation; when using displacement, it represents the
    mean movement direction. When no valid headings exist, or when the summed
    heading vector has zero magnitude (for example exact cancellation), the
    returned angle is NaN.

    When ``validate_ap=True`` and ``body_axis_keypoints`` is provided,
    anterior-posterior validation is run per individual and the result is
    stored in ``polarization.attrs["ap_validation_result"]``.

    Examples
    --------
    Compute orientation polarization from body-axis keypoints:

    >>> polarization = compute_polarization(
    ...     ds.position,
    ...     body_axis_keypoints=("tail_base", "neck"),
    ... )

    Compute heading polarization from displacement (pre-select keypoint):

    >>> polarization = compute_polarization(
    ...     ds.position.sel(keypoints="thorax")
    ... )

    If multiple keypoints exist and none is selected, the first is used:

    >>> polarization = compute_polarization(ds.position)

    Return orientation polarization with mean body orientation angle:

    >>> polarization, mean_angle = compute_polarization(
    ...     ds.position,
    ...     body_axis_keypoints=("tail_base", "neck"),
    ...     return_angle=True,
    ... )

    Return heading polarization with mean movement direction angle (radians):

    >>> polarization, mean_angle = compute_polarization(
    ...     ds.position.sel(keypoints="thorax"),
    ...     return_angle=True,
    ... )

    Return heading polarization with mean movement direction angle (degrees):

    >>> polarization, mean_angle = compute_polarization(
    ...     ds.position.sel(keypoints="thorax"),
    ...     return_angle=True,
    ...     in_degrees=True,
    ... )

    If multiple keypoints exist, first is used; also return mean angle:

    >>> polarization, mean_angle = compute_polarization(
    ...     ds.position,
    ...     return_angle=True,
    ... )

    Run AP validation while computing body-axis polarization:

    >>> polarization = compute_polarization(
    ...     ds.position,
    ...     body_axis_keypoints=("tail_base", "neck"),
    ...     validate_ap=True,
    ... )

    """
    _validate_type_data_array(data)
    normalized_keypoints = _validate_position_data(
        data=data,
        body_axis_keypoints=body_axis_keypoints,
    )

    ap_validation_result = None
    if normalized_keypoints is not None:
        if validate_ap:
            ap_validation_result = _run_ap_validation(
                data, normalized_keypoints, ap_validation_config
            )
        heading_vectors = _compute_heading_from_keypoints(
            data=data,
            body_axis_keypoints=normalized_keypoints,
        )
    else:
        heading_vectors = _compute_heading_from_velocity(
            data=data,
            displacement_frames=displacement_frames,
        )

    polarization = _compute_polarization_from_headings(heading_vectors)

    if ap_validation_result is not None:
        polarization.attrs["ap_validation_result"] = ap_validation_result

    if not return_angle:
        return polarization

    mean_angle = _compute_mean_angle(heading_vectors, in_degrees)
    return polarization, mean_angle


def _compute_polarization_from_headings(
    heading_vectors: xr.DataArray,
) -> xr.DataArray:
    """Compute polarization magnitude from heading vectors."""
    heading = _select_space(heading_vectors)
    unit_headings = convert_to_unit(heading)
    valid_mask = ~unit_headings.isnull().any(dim="space")

    vector_sum = unit_headings.sum(dim="individuals", skipna=True)
    sum_magnitude = compute_norm(vector_sum)
    n_valid = valid_mask.sum(dim="individuals")

    polarization = xr.where(
        n_valid > 0,
        sum_magnitude / n_valid,
        np.nan,
    ).clip(min=0.0, max=1.0)
    return polarization.rename("polarization")


def _compute_mean_angle(
    heading_vectors: xr.DataArray,
    in_degrees: bool = False,
) -> xr.DataArray:
    """Compute mean heading angle from heading vectors."""
    heading = _select_space(heading_vectors)
    unit_headings = convert_to_unit(heading)
    valid_mask = ~unit_headings.isnull().any(dim="space")

    vector_sum = unit_headings.sum(dim="individuals", skipna=True)
    sum_magnitude = compute_norm(vector_sum)
    n_valid = valid_mask.sum(dim="individuals")

    mean_unit_vector = vector_sum / sum_magnitude
    reference = np.array([1, 0])
    angle_defined = (n_valid > 0) & (sum_magnitude > _ANGLE_EPS)
    mean_angle = xr.where(
        angle_defined,
        compute_signed_angle_2d(
            mean_unit_vector, reference, v_as_left_operand=True
        ),
        np.nan,
    )
    if in_degrees:
        mean_angle = np.rad2deg(mean_angle)
    return mean_angle.rename("mean_angle")


def _run_ap_validation(
    data: xr.DataArray,
    normalized_keypoints: tuple[Hashable, Hashable],
    ap_validation_config: dict[str, Any] | None,
) -> dict:
    """Run AP validation across all individuals, select best by R*M.

    Each individual is validated independently using the supplied keypoint
    pair. R*M (resultant_length × vote_margin) is computed per individual
    and depends only on the individual's motion and body shape, not on
    the input pair. The best individual is the one with the highest R*M.
    """
    config = (
        _ValidateAPConfig(**ap_validation_config)
        if ap_validation_config is not None
        else None
    )

    if "individuals" not in data.dims:
        single_result = _validate_ap(
            data,
            from_node=normalized_keypoints[0],
            to_node=normalized_keypoints[1],
            config=config,
            verbose=False,
        )
        return {"all_results": [single_result], "best_idx": 0}

    individuals = list(data.coords["individuals"].values)
    all_results = []
    for individual in individuals:
        result = _validate_ap(
            data.sel(individuals=individual),
            from_node=normalized_keypoints[0],
            to_node=normalized_keypoints[1],
            config=config,
            verbose=False,
        )
        result["individual"] = individual
        all_results.append(result)

    best_idx = _find_best_individual_by_rxm(all_results)
    return {"all_results": all_results, "best_idx": best_idx}


def _find_best_individual_by_rxm(all_results: list[dict]) -> int:
    """Return index of the individual with highest R*M score."""
    best_idx = -1
    best_rxm = -1.0
    for i, result in enumerate(all_results):
        if not result["success"]:
            continue
        rxm = result["resultant_length"] * result["vote_margin"]
        if rxm > best_rxm:
            best_rxm = rxm
            best_idx = i
    return best_idx


def _compute_heading_from_keypoints(
    data: xr.DataArray,
    body_axis_keypoints: tuple[Hashable, Hashable],
) -> xr.DataArray:
    """Compute heading vectors from two keypoints (origin to target)."""
    origin, target = body_axis_keypoints
    heading = data.sel(keypoints=target, drop=True) - data.sel(
        keypoints=origin,
        drop=True,
    )
    return heading


def _compute_heading_from_velocity(
    data: xr.DataArray,
    displacement_frames: int = 1,
) -> xr.DataArray:
    """Compute heading vectors from displacement direction."""
    _validate_displacement_frames(displacement_frames)

    position = data
    if "keypoints" in data.dims:
        if data.sizes["keypoints"] < 1:
            raise ValueError(
                "data.keypoints must contain at least one keypoint."
            )
        position = data.isel(keypoints=0, drop=True)

        if "keypoints" in data.coords and data.coords["keypoints"].size > 0:
            logger.info(
                "Using keypoint '%s' for displacement-based heading.",
                data.coords["keypoints"].values[0],
            )
        else:
            logger.info(
                "Using keypoint index 0 for displacement-based heading."
            )

    displacement = position - position.shift(time=displacement_frames)
    return displacement


def _select_space(data: xr.DataArray) -> xr.DataArray:
    """Return data with standard dim order, preserving all spatial coords."""
    return data.transpose("time", "space", "individuals")


def _validate_position_data(
    data: xr.DataArray,
    body_axis_keypoints: tuple[Hashable, Hashable] | None,
) -> tuple[Hashable, Hashable] | None:
    """Validate the input array and normalize ``body_axis_keypoints``."""
    validate_dims_coords(
        data,
        {
            "time": [],
            "space": [],
            "individuals": [],
        },
    )

    allowed_dims = {"time", "space", "individuals", "keypoints"}
    unexpected_dims = set(data.dims) - allowed_dims
    if unexpected_dims:
        raise ValueError(
            f"data contains unsupported dimension(s): "
            f"{sorted(str(d) for d in unexpected_dims)}"
        )

    if "space" not in data.coords:
        raise ValueError(
            "data must have coordinate labels for the 'space' dimension."
        )

    space_labels = set(data.coords["space"].values.tolist())
    if not {"x", "y"}.issubset(space_labels):
        raise ValueError(
            "data.space must include coordinate labels 'x' and 'y'."
        )

    if body_axis_keypoints is None:
        return None

    origin, target = _normalize_body_axis_keypoints(body_axis_keypoints)

    if "keypoints" not in data.dims:
        raise ValueError(
            "body_axis_keypoints requires a 'keypoints' dimension in data."
        )

    validate_dims_coords(data, {"keypoints": [origin, target]})
    return origin, target


def _normalize_body_axis_keypoints(
    body_axis_keypoints: tuple[Hashable, Hashable] | Any,
) -> tuple[Hashable, Hashable]:
    """Validate and normalize the keypoint pair."""
    if isinstance(body_axis_keypoints, (str, bytes)):
        raise TypeError(
            "body_axis_keypoints must be an iterable of exactly two "
            "keypoint names."
        )

    try:
        origin, target = body_axis_keypoints
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "body_axis_keypoints must be an iterable of exactly two "
            "keypoint names."
        ) from exc

    for keypoint in (origin, target):
        if not isinstance(keypoint, Hashable):
            raise TypeError("Each body axis keypoint must be hashable.")

    if origin == target:
        raise ValueError(
            "body_axis_keypoints must contain two distinct keypoint names."
        )

    return origin, target


def _validate_displacement_frames(displacement_frames: int) -> None:
    """Validate the displacement window."""
    if isinstance(displacement_frames, (bool, np.bool_)) or not isinstance(
        displacement_frames,
        (int, np.integer),
    ):
        raise TypeError("displacement_frames must be a positive integer.")

    if displacement_frames < 1:
        raise ValueError("displacement_frames must be >= 1.")


def _validate_type_data_array(data: xr.DataArray) -> None:
    """Validate that the input is an xarray.DataArray."""
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Input data must be an xarray.DataArray, but got {type(data)}."
        )


# Helper functions for _validate_ap


def _compute_tiered_validity(
    keypoints: np.ndarray,
    min_valid_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tiered validity masks for each frame.

    Parameters
    ----------
    keypoints : np.ndarray
        Keypoint positions with shape (n_frames, n_keypoints, 2).
    min_valid_frac : float
        Minimum fraction of keypoints required for tier-1 validity.

    Returns
    -------
    tier1_valid : np.ndarray
        Boolean array of shape (n_frames,) indicating tier-1 valid frames.
        A frame is tier-1 valid if at least min_valid_frac of keypoints
        are present AND at least 2 keypoints are present.
    tier2_valid : np.ndarray
        Boolean array of shape (n_frames,) indicating tier-2 valid frames.
        A frame is tier-2 valid if all keypoints are present.
    frac_present : np.ndarray
        Array of shape (n_frames,) with fraction of keypoints present.

    """
    n_frames, n_keypoints, _ = keypoints.shape

    # A keypoint is present if neither x nor y is NaN
    # Shape: (n_frames, n_keypoints)
    keypoint_present = ~np.any(np.isnan(keypoints), axis=2)

    # Count present keypoints per frame
    n_present = np.sum(keypoint_present, axis=1)
    frac_present = n_present / n_keypoints

    # Tier-2: all keypoints present
    tier2_valid = n_present == n_keypoints

    # Tier-1: at least min_valid_frac present AND at least 2 present
    tier1_valid = (frac_present >= min_valid_frac) & (n_present >= 2)

    return tier1_valid, tier2_valid, frac_present


def _compute_bbox_centroid(
    keypoints: np.ndarray,
    tier1_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bounding-box centroids for tier-1 valid frames.

    The bounding-box centroid is the midpoint of the axis-aligned bounding
    box enclosing all present keypoints. This is density-invariant, unlike
    the arithmetic mean.

    Parameters
    ----------
    keypoints : np.ndarray
        Keypoint positions with shape (n_frames, n_keypoints, 2).
    tier1_valid : np.ndarray
        Boolean array of shape (n_frames,) indicating tier-1 valid frames.

    Returns
    -------
    bbox_centroids : np.ndarray
        Array of shape (n_frames, 2) with bounding-box centroids.
        NaN for non-tier-1-valid frames.
    arith_centroids : np.ndarray
        Array of shape (n_frames, 2) with arithmetic-mean centroids.
        NaN for non-tier-1-valid frames. Used for diagnostic comparison.
    centroid_discrepancy : np.ndarray
        Array of shape (n_frames,) with normalized discrepancy between
        bbox and arithmetic centroids (distance / bbox_diagonal).
        NaN for non-tier-1-valid frames.

    """
    n_frames = keypoints.shape[0]

    bbox_centroids = np.full((n_frames, 2), np.nan)
    arith_centroids = np.full((n_frames, 2), np.nan)
    centroid_discrepancy = np.full(n_frames, np.nan)

    for f in range(n_frames):
        if not tier1_valid[f]:
            continue

        kp_f = keypoints[f]  # (n_keypoints, 2)

        # Find present keypoints (no NaN in either coordinate)
        present_mask = ~np.any(np.isnan(kp_f), axis=1)
        kp_present = kp_f[present_mask]

        # Bounding-box centroid
        bbox_min = np.min(kp_present, axis=0)
        bbox_max = np.max(kp_present, axis=0)
        bbox_centroids[f] = (bbox_min + bbox_max) / 2

        # Arithmetic-mean centroid
        arith_centroids[f] = np.mean(kp_present, axis=0)

        # Centroid discrepancy: distance normalized by bbox diagonal
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        if bbox_diag > 0:
            discrepancy = np.linalg.norm(
                bbox_centroids[f] - arith_centroids[f]
            )
            centroid_discrepancy[f] = discrepancy / bbox_diag
        else:
            centroid_discrepancy[f] = 0.0

    return bbox_centroids, arith_centroids, centroid_discrepancy


def _compute_frame_velocities(
    bbox_centroids: np.ndarray,
    tier1_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frame-to-frame centroid velocities and speeds.

    A velocity is valid only when both adjacent frames are tier-1 valid.

    Parameters
    ----------
    bbox_centroids : np.ndarray
        Array of shape (n_frames, 2) with bounding-box centroids.
    tier1_valid : np.ndarray
        Boolean array of shape (n_frames,) indicating tier-1 valid frames.

    Returns
    -------
    velocities : np.ndarray
        Array of shape (n_frames - 1, 2) with velocity vectors.
        Invalid velocities are NaN.
    speeds : np.ndarray
        Array of shape (n_frames - 1,) with speed scalars.
        Invalid speeds are NaN.

    """
    velocities = np.diff(bbox_centroids, axis=0)  # (n_frames - 1, 2)

    # Velocity valid only if both adjacent frames are tier-1 valid
    speed_valid = tier1_valid[:-1] & tier1_valid[1:]

    # Mask invalid velocities
    velocities[~speed_valid] = np.nan

    speeds = np.linalg.norm(velocities, axis=1)

    return velocities, speeds


def _compute_sliding_window_medians(
    speeds: np.ndarray,
    window_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute median speeds for sliding windows.

    A window is valid only when every speed sample in that window is valid
    (non-NaN), ensuring strict NaN-free content.

    Parameters
    ----------
    speeds : np.ndarray
        Array of shape (n_speed_samples,) with speed values.
    window_len : int
        Number of speed samples per sliding window.
    stride : int
        Step size between consecutive window start positions.

    Returns
    -------
    window_starts : np.ndarray
        Array of window start indices (0-indexed).
    window_medians : np.ndarray
        Median speed for each window. NaN for invalid windows.
    window_all_valid : np.ndarray
        Boolean array indicating which windows are fully valid.

    """
    num_speed = len(speeds)
    window_starts = np.arange(0, num_speed - window_len + 1, stride)
    num_windows = len(window_starts)

    window_medians = np.full(num_windows, np.nan)
    window_all_valid = np.zeros(num_windows, dtype=bool)

    for k in range(num_windows):
        s = window_starts[k]
        e = s + window_len
        w = speeds[s:e]

        # Window valid only if all samples are non-NaN
        if np.all(~np.isnan(w)):
            window_all_valid[k] = True
            window_medians[k] = np.median(w)

    return window_starts, window_medians, window_all_valid


def _detect_high_motion_windows(
    window_medians: np.ndarray,
    window_all_valid: np.ndarray,
    pct_thresh: float,
) -> np.ndarray:
    """Identify high-motion windows based on percentile threshold.

    Parameters
    ----------
    window_medians : np.ndarray
        Median speed for each window.
    window_all_valid : np.ndarray
        Boolean array indicating which windows are fully valid.
    pct_thresh : float
        Percentile threshold (0-100) for high-motion classification.

    Returns
    -------
    high_motion : np.ndarray
        Boolean array indicating high-motion windows.

    """
    valid_medians = window_medians[window_all_valid]
    if len(valid_medians) == 0:
        return np.zeros(len(window_medians), dtype=bool)

    thresh = np.percentile(valid_medians, pct_thresh)
    high_motion = window_all_valid & (window_medians >= thresh)

    return high_motion


def _detect_runs(
    high_motion: np.ndarray,
    min_run_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect runs of consecutive high-motion windows.

    A run is a maximal sequence of consecutively indexed qualifying windows.

    Parameters
    ----------
    high_motion : np.ndarray
        Boolean array indicating high-motion windows.
    min_run_len : int
        Minimum number of consecutive qualifying windows for a valid run.

    Returns
    -------
    run_starts : np.ndarray
        Start indices of valid runs.
    run_ends : np.ndarray
        End indices (inclusive) of valid runs.
    run_lengths : np.ndarray
        Length of each valid run.

    """
    padded = np.concatenate([[False], high_motion, [False]])
    d = np.diff(padded.astype(int))

    # Find run boundaries
    run_starts_all = np.where(d == 1)[0]
    run_ends_all = np.where(d == -1)[0] - 1
    run_lengths_all = run_ends_all - run_starts_all + 1

    # Filter by minimum run length
    valid_mask = run_lengths_all >= min_run_len
    run_starts = run_starts_all[valid_mask]
    run_ends = run_ends_all[valid_mask]
    run_lengths = run_lengths_all[valid_mask]

    return run_starts, run_ends, run_lengths


def _convert_runs_to_segments(
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    window_starts: np.ndarray,
    window_len: int,
) -> np.ndarray:
    """Convert window runs to frame segments.

    Each run is converted to a frame interval spanning from the start frame
    of the first window to the end frame of the last window.

    Parameters
    ----------
    run_starts : np.ndarray
        Start indices of valid runs (indices into window arrays).
    run_ends : np.ndarray
        End indices (inclusive) of valid runs.
    window_starts : np.ndarray
        Start frame indices for each window.
    window_len : int
        Length of each window in frames.

    Returns
    -------
    segments_raw : np.ndarray
        Array of shape (n_runs, 2) with [frame_start, frame_end] for each run.

    """
    n_runs = len(run_starts)
    segments_raw = np.zeros((n_runs, 2), dtype=int)

    for j in range(n_runs):
        s_idx = run_starts[j]
        e_idx = run_ends[j]
        frame_start = window_starts[s_idx]
        frame_end = window_starts[e_idx] + window_len
        segments_raw[j] = [frame_start, frame_end]

    return segments_raw


def _merge_segments(segments_raw: np.ndarray) -> np.ndarray:
    """Merge overlapping or abutting frame segments.

    Segments are first sorted by start frame, then merged if they overlap
    or abut (next start <= current end + 1).

    Parameters
    ----------
    segments_raw : np.ndarray
        Array of shape (n_segments, 2) with [frame_start, frame_end].

    Returns
    -------
    segments : np.ndarray
        Array of merged non-overlapping segments.

    """
    if len(segments_raw) == 0:
        return segments_raw

    # Sort by start frame
    sorted_idx = np.argsort(segments_raw[:, 0])
    segments_sorted = segments_raw[sorted_idx]

    merged = [segments_sorted[0].tolist()]

    for j in range(1, len(segments_sorted)):
        next_seg = segments_sorted[j]
        curr_seg = merged[-1]

        # Merge if overlapping or abutting
        if next_seg[0] <= curr_seg[1] + 1:
            merged[-1][1] = max(curr_seg[1], next_seg[1])
        else:
            merged.append(next_seg.tolist())

    return np.array(merged, dtype=int)


def _filter_segments_tier2(
    segments: np.ndarray,
    tier2_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter segment frames to retain only tier-2 valid frames.

    Parameters
    ----------
    segments : np.ndarray
        Array of shape (n_segments, 2) with [frame_start, frame_end].
    tier2_valid : np.ndarray
        Boolean array of shape (n_frames,) indicating tier-2 valid frames.

    Returns
    -------
    selected_frames : np.ndarray
        Array of tier-2 valid frame indices within segments.
    selected_seg_id : np.ndarray
        Segment ID (0-indexed) for each selected frame.

    """
    # Collect all unique frames from all segments
    all_segment_frames: list[int] = []
    for k in range(len(segments)):
        frame_start, frame_end = segments[k]
        seg_frames = np.arange(frame_start, frame_end + 1)
        all_segment_frames.extend(seg_frames)

    segment_frames_all = np.unique(all_segment_frames)

    tier2_mask = tier2_valid[segment_frames_all]
    selected_frames = segment_frames_all[tier2_mask]

    # Assign each selected frame to its segment
    num_selected = len(selected_frames)
    selected_seg_id = np.zeros(num_selected, dtype=int)

    for j in range(num_selected):
        f = selected_frames[j]
        for k in range(len(segments)):
            if segments[k, 0] <= f <= segments[k, 1]:
                selected_seg_id[j] = k
                break

    return selected_frames, selected_seg_id


def _build_centered_skeletons(
    keypoints: np.ndarray,
    selected_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build centroid-centered skeletons for selected frames.

    Uses bounding-box centroid for centering, consistent with the
    segmentation centroid.

    Parameters
    ----------
    keypoints : np.ndarray
        Keypoint positions with shape (n_frames, n_keypoints, 2).
    selected_frames : np.ndarray
        Array of selected frame indices.

    Returns
    -------
    selected_centroids : np.ndarray
        Array of shape (num_selected, 2) with bounding-box centroids.
    centered_skeletons : np.ndarray
        Array of shape (num_selected, n_keypoints, 2) with
        centroid-centered skeleton coordinates.

    """
    num_selected = len(selected_frames)
    n_keypoints = keypoints.shape[1]

    selected_centroids = np.zeros((num_selected, 2))
    centered_skeletons = np.zeros((num_selected, n_keypoints, 2))

    for j in range(num_selected):
        f = selected_frames[j]
        kp_f = keypoints[f]  # (n_keypoints, 2) - all present for tier-2

        # Bounding-box centroid
        bbox_min = np.min(kp_f, axis=0)
        bbox_max = np.max(kp_f, axis=0)
        centroid_f = (bbox_min + bbox_max) / 2

        selected_centroids[j] = centroid_f
        centered_skeletons[j] = kp_f - centroid_f

    return selected_centroids, centered_skeletons


def _compute_pairwise_rmsd(centered_skeletons: np.ndarray) -> np.ndarray:
    """Compute pairwise RMSD between all centered skeletons.

    RMSD is computed as the square root of the mean of squared entry-wise
    differences between flattened skeleton vectors.

    Parameters
    ----------
    centered_skeletons : np.ndarray
        Array of shape (num_selected, n_keypoints, 2).

    Returns
    -------
    rmsd_matrix : np.ndarray
        Symmetric matrix of shape (num_selected, num_selected) with
        pairwise RMSD values. Diagonal is zero.

    """
    num_selected = len(centered_skeletons)

    skel_flat = centered_skeletons.reshape(num_selected, -1)

    rmsd_matrix = np.zeros((num_selected, num_selected))

    for i in range(num_selected):
        for j in range(i + 1, num_selected):
            d = skel_flat[i] - skel_flat[j]
            rmsd_val = np.sqrt(np.mean(d**2))
            rmsd_matrix[i, j] = rmsd_val
            rmsd_matrix[j, i] = rmsd_val

    return rmsd_matrix


def _compute_postural_variance_ratio(
    rmsd_matrix: np.ndarray,
    selected_seg_id: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, bool]:
    """Compute the between/within segment RMSD variance ratio.

    Parameters
    ----------
    rmsd_matrix : np.ndarray
        Pairwise RMSD matrix of shape (num_selected, num_selected).
    selected_seg_id : np.ndarray
        Segment ID for each selected frame.

    Returns
    -------
    var_ratio : float
        Ratio of between-segment to within-segment RMSD variance.
        Returns 0.0 if either distribution is empty or within variance is 0.
    within_rmsds : np.ndarray
        Array of within-segment RMSD values.
    between_rmsds : np.ndarray
        Array of between-segment RMSD values.
    var_ratio_override : bool
        True if variance ratio was set to 0 due to edge cases.

    """
    num_selected = len(selected_seg_id)
    within_rmsds_list: list[float] = []
    between_rmsds_list: list[float] = []

    for i in range(num_selected):
        for j in range(i + 1, num_selected):
            if selected_seg_id[i] == selected_seg_id[j]:
                within_rmsds_list.append(rmsd_matrix[i, j])
            else:
                between_rmsds_list.append(rmsd_matrix[i, j])

    within_rmsds = np.array(within_rmsds_list)
    between_rmsds = np.array(between_rmsds_list)

    # Compute variance ratio with edge case handling
    var_ratio_override = False
    if (
        len(within_rmsds) > 0
        and len(between_rmsds) > 0
        and np.var(within_rmsds) > 0
    ):
        var_ratio = np.var(between_rmsds) / np.var(within_rmsds)
    else:
        var_ratio = 0.0
        var_ratio_override = True

    return var_ratio, within_rmsds, between_rmsds, var_ratio_override


def _kmedoids(
    data: np.ndarray,
    k: int,
    max_iter: int = 100,
    n_init: int = 5,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Perform k-medoids clustering.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, n_features).
    k : int
        Number of clusters.
    max_iter : int, default=100
        Maximum number of iterations.
    n_init : int, default=5
        Number of random initializations.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample (0-indexed).
    medoid_indices : np.ndarray
        Indices of medoid samples.
    inertia : float
        Sum of distances from samples to their medoids.

    """
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(random_state)
    n_samples = len(data)

    dist_matrix = cdist(data, data, metric="euclidean")

    best_labels: np.ndarray | None = None
    best_medoids: np.ndarray | None = None
    best_inertia = np.inf

    for _ in range(n_init):
        medoids = rng.choice(n_samples, size=k, replace=False)

        for _ in range(max_iter):
            distances_to_medoids = dist_matrix[:, medoids]
            labels = np.argmin(distances_to_medoids, axis=1)

            # Update medoids
            new_medoids = np.zeros(k, dtype=int)
            for cluster in range(k):
                cluster_mask = labels == cluster
                if not np.any(cluster_mask):
                    # Empty cluster - keep old medoid
                    new_medoids[cluster] = medoids[cluster]
                    continue

                cluster_indices = np.where(cluster_mask)[0]
                # Find point that minimizes sum of distances within cluster
                cluster_dists = dist_matrix[
                    np.ix_(cluster_indices, cluster_indices)
                ]
                total_dists = np.sum(cluster_dists, axis=1)
                best_idx = np.argmin(total_dists)
                new_medoids[cluster] = cluster_indices[best_idx]

            if np.array_equal(np.sort(medoids), np.sort(new_medoids)):
                break
            medoids = new_medoids

        distances_to_medoids = dist_matrix[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)
        inertia = np.sum(distances_to_medoids[np.arange(n_samples), labels])

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_medoids = medoids.copy()

    # These are guaranteed to be set after at least one iteration
    assert best_labels is not None and best_medoids is not None
    return best_labels, best_medoids, best_inertia


def _silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels for each sample.

    Returns
    -------
    score : float
        Mean silhouette score across all samples.
        Returns 0.0 if clustering is degenerate.

    """
    from scipy.spatial.distance import cdist

    n_samples = len(data)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1 or n_clusters >= n_samples:
        return 0.0

    dist_matrix = cdist(data, data, metric="euclidean")

    silhouette_vals = np.zeros(n_samples)

    for i in range(n_samples):
        own_cluster = labels[i]
        own_mask = labels == own_cluster

        # a(i) = mean distance to points in same cluster
        if np.sum(own_mask) > 1:
            a_i = np.mean(
                dist_matrix[i, own_mask & (np.arange(n_samples) != i)]
            )
        else:
            a_i = 0.0

        # b(i) = min over other clusters of mean distance to that cluster
        b_i = np.inf
        for cluster in unique_labels:
            if cluster == own_cluster:
                continue
            cluster_mask = labels == cluster
            if np.any(cluster_mask):
                mean_dist = np.mean(dist_matrix[i, cluster_mask])
                b_i = min(b_i, mean_dist)

        if b_i == np.inf:
            silhouette_vals[i] = 0.0
        else:
            silhouette_vals[i] = (
                (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
            )

    return float(np.mean(silhouette_vals))


def _perform_postural_clustering(
    centered_skeletons: np.ndarray,
    max_clusters: int,
    min_silhouette: float = 0.2,
) -> tuple[np.ndarray, int, int, float, list[tuple[int, float]]]:
    """Perform postural clustering using k-medoids with silhouette selection.

    Parameters
    ----------
    centered_skeletons : np.ndarray
        Array of shape (num_selected, n_keypoints, 2).
    max_clusters : int
        Maximum number of clusters to evaluate.
    min_silhouette : float, default=0.2
        Minimum silhouette score to accept clustering.

    Returns
    -------
    cluster_labels : np.ndarray
        Cluster labels for each frame (0-indexed).
    num_clusters : int
        Number of clusters (1 if clustering not accepted).
    primary_cluster : int
        Index of largest cluster (0-indexed).
    best_silhouette : float
        Best silhouette score achieved.
    silhouette_scores : list of (k, score)
        Silhouette scores for each k evaluated.

    """
    num_selected = len(centered_skeletons)
    skel_flat = centered_skeletons.reshape(num_selected, -1)

    best_k = 1
    best_sil = -np.inf
    silhouette_scores = []

    # Evaluate k from 2 to max_clusters (capped at num_selected // 2)
    max_k = min(max_clusters, num_selected // 2)

    for k in range(2, max_k + 1):
        try:
            labels, _, _ = _kmedoids(skel_flat, k, n_init=5)
            sil = _silhouette_score(skel_flat, labels)
            silhouette_scores.append((k, sil))

            if sil > best_sil:
                best_sil = sil
                best_k = k
        except Exception:
            silhouette_scores.append((k, np.nan))

    # Accept clustering only if best_sil > min_silhouette
    if best_k > 1 and best_sil > min_silhouette:
        # Re-run with more initializations for final result
        cluster_labels, _, _ = _kmedoids(skel_flat, best_k, n_init=10)
        num_clusters = best_k

        # Primary cluster = largest
        cluster_counts = np.bincount(cluster_labels, minlength=num_clusters)
        primary_cluster = int(np.argmax(cluster_counts))
    else:
        cluster_labels = np.zeros(num_selected, dtype=int)
        num_clusters = 1
        primary_cluster = 0

    return (
        cluster_labels,
        num_clusters,
        primary_cluster,
        best_sil,
        silhouette_scores,
    )


def _compute_cluster_velocities(
    selected_frames: np.ndarray,
    selected_seg_id: np.ndarray,
    cluster_mask: np.ndarray,
    segments: np.ndarray,
    bbox_centroids: np.ndarray,
) -> np.ndarray:
    """Compute velocities between adjacent consecutive frames.

    Only considers frames in the same segment and cluster. Frame pairs
    where both frames are consecutive (frame[i] == frame[i-1] + 1),
    in the same segment, and in the same cluster contribute a velocity
    vector.
    This prevents spanning temporal gaps or mixing postures across clusters.

    Returns
    -------
    np.ndarray
        Array of shape (n_velocities, 2). Empty (0, 2) if no valid pairs.

    """
    frames_c = selected_frames[cluster_mask]
    seg_ids_c = selected_seg_id[cluster_mask]
    velocities_list: list[np.ndarray] = []

    for seg_k in range(len(segments)):
        seg_mask = seg_ids_c == seg_k
        seg_frames = np.sort(frames_c[seg_mask])
        for fi in range(1, len(seg_frames)):
            if seg_frames[fi] != seg_frames[fi - 1] + 1:
                continue
            curr_frame = seg_frames[fi]
            prev_frame = seg_frames[fi - 1]
            v = bbox_centroids[curr_frame] - bbox_centroids[prev_frame]
            if np.all(~np.isnan(v)):
                velocities_list.append(v)

    return np.array(velocities_list) if velocities_list else np.zeros((0, 2))


def _infer_anterior_from_velocities(
    velocities: np.ndarray,
    PC1: np.ndarray,
) -> dict:
    """Infer anterior direction from velocity projections onto PC1.

    Uses strict majority vote on PC1 projection signs: anterior = +PC1
    if n_positive > n_negative, else −PC1 (ties default to −PC1).

    Also computes circular statistics on velocity angles:
    - resultant_length R = √(C² + S²) where C = mean(cos θ), S = mean(sin θ)
    - vote_margin M = |n₊ − n₋| / (n₊ + n₋)

    Returns dict with resultant_length, circ_mean_dir, vel_projs_pc1,
    num_positive, num_negative, vote_margin, anterior_sign.
    """
    result: dict = {
        "resultant_length": 0.0,
        "circ_mean_dir": np.nan,
        "vel_projs_pc1": np.array([]),
        "num_positive": 0,
        "num_negative": 0,
        "vote_margin": 0.0,
        "anterior_sign": -1,
    }
    if len(velocities) == 0:
        return result

    vel_angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    circ_C = np.mean(np.cos(vel_angles))
    circ_S = np.mean(np.sin(vel_angles))
    result["resultant_length"] = np.sqrt(circ_C**2 + circ_S**2)
    result["circ_mean_dir"] = np.arctan2(circ_S, circ_C)

    vel_projs = velocities @ PC1
    num_pos = int(np.sum(vel_projs > 0))
    num_neg = int(np.sum(vel_projs < 0))
    result["vel_projs_pc1"] = vel_projs
    result["num_positive"] = num_pos
    result["num_negative"] = num_neg
    result["vote_margin"] = abs(num_pos - num_neg) / max(num_pos + num_neg, 1)
    result["anterior_sign"] = +1 if num_pos > num_neg else -1
    return result


def _compute_cluster_pca_and_anterior(
    centered_skeletons: np.ndarray,
    cluster_mask: np.ndarray,
    selected_frames: np.ndarray,
    selected_seg_id: np.ndarray,
    segments: np.ndarray,
    bbox_centroids: np.ndarray,
) -> dict:
    """Compute SVD-based PCA and velocity-based anterior inference.

    Performs inference for one cluster.

    Performs SVD on the cluster's average centered skeleton to extract PC1/PC2,
    applies the geometric sign convention, then infers the anterior direction
    via velocity voting on centroid displacements projected onto PC1.

    Returns
    -------
    dict
        Keys: valid, n_frames, avg_skeleton, valid_shape_rows,
        PC1, PC2, anterior_sign, vote_margin, resultant_length,
        circ_mean_dir, velocities, vel_projs_pc1, and others.

    """
    n_keypoints = centered_skeletons.shape[1]
    n_c = int(np.sum(cluster_mask))

    result: dict = {
        "valid": False,
        "n_frames": n_c,
        "avg_skeleton": np.full((n_keypoints, 2), np.nan),
        "valid_shape_rows": np.zeros(n_keypoints, dtype=bool),
        "PC1": np.array([1.0, 0.0]),
        "PC2": np.array([0.0, 1.0]),
        "proj_pc1": np.full(n_keypoints, np.nan),
        "proj_pc2": np.full(n_keypoints, np.nan),
        "anterior_sign": -1,
        "num_positive": 0,
        "num_negative": 0,
        "vote_margin": 0.0,
        "resultant_length": 0.0,
        "circ_mean_dir": np.nan,
        "velocities": np.zeros((0, 2)),
        "vel_projs_pc1": np.array([]),
    }

    if n_c == 0:
        return result

    skels_c = centered_skeletons[cluster_mask]
    avg_skel_c = np.mean(skels_c, axis=0)
    valid_shape_rows = ~np.any(np.isnan(avg_skel_c), axis=1)

    if np.sum(valid_shape_rows) < 2:
        return result

    result["avg_skeleton"] = avg_skel_c
    result["valid_shape_rows"] = valid_shape_rows

    # SVD on valid (non-NaN) rows of the average centered skeleton
    D_valid = avg_skel_c[valid_shape_rows]
    _U, _S, Vt = np.linalg.svd(D_valid, full_matrices=False)
    PC1 = Vt[0]
    PC2 = Vt[1] if len(Vt) > 1 else np.array([0.0, 1.0])

    # Geometric sign convention (reproducible across runs, decoupled
    # from anatomical AP assignment which is determined by velocity voting):
    #   PC1 flipped so y-component >= 0
    #   PC2 flipped so x-component >= 0

    if PC1[1] < 0:
        PC1 = -PC1
    if PC2[0] < 0:
        PC2 = -PC2

    result["PC1"] = PC1
    result["PC2"] = PC2

    proj_pc1 = np.full(n_keypoints, np.nan)
    proj_pc2 = np.full(n_keypoints, np.nan)
    proj_pc1[valid_shape_rows] = avg_skel_c[valid_shape_rows] @ PC1
    proj_pc2[valid_shape_rows] = avg_skel_c[valid_shape_rows] @ PC2
    result["proj_pc1"] = proj_pc1
    result["proj_pc2"] = proj_pc2

    velocities = _compute_cluster_velocities(
        selected_frames,
        selected_seg_id,
        cluster_mask,
        segments,
        bbox_centroids,
    )
    result["velocities"] = velocities
    result.update(_infer_anterior_from_velocities(velocities, PC1))
    result["valid"] = True
    return result


def _compute_node_projections(
    report: _APNodePairReport,
    avg_skeleton: np.ndarray,
    PC1: np.ndarray,
    anterior_sign: int,
    valid_shape_rows: np.ndarray,
    from_node: int,
    to_node: int,
) -> None:
    """Compute raw PC1, AP-oriented, and lateral projections.

    Computes projections for all valid keypoints.

    Populates the report's coordinate arrays and determines:
    - pc1_coords: raw projection onto PC1 (sign-convention only)
    - ap_coords: projection onto anterior_sign × PC1 (positive = more
      anterior)
    - lateral_offsets: unsigned distance from the AP axis
    - midpoint_pc1: average of min/max PC1 projections (AP reference point)
    - input_pair_order_matches_inference: True if from_node's AP coord <
      to_node's
    """
    pc1 = PC1 / np.linalg.norm(PC1)
    # AP unit vector: anterior_sign * PC1, so positive projection = anterior
    e_ap = anterior_sign * pc1
    # Lateral unit vector: 90° CCW rotation of e_ap
    e_lat = np.array([-e_ap[1], e_ap[0]])

    D_valid = avg_skeleton[valid_shape_rows]
    report.pc1_coords[valid_shape_rows] = D_valid @ pc1
    report.ap_coords[valid_shape_rows] = D_valid @ e_ap
    report.lateral_offsets[valid_shape_rows] = np.abs(D_valid @ e_lat)

    if valid_shape_rows[from_node] and valid_shape_rows[to_node]:
        report.input_pair_order_matches_inference = (
            report.ap_coords[from_node] < report.ap_coords[to_node]
        )

    proj_pc1_valid = report.pc1_coords[valid_shape_rows]
    report.pc1_min = float(np.min(proj_pc1_valid))
    report.pc1_max = float(np.max(proj_pc1_valid))
    report.midpoint_pc1 = (report.pc1_min + report.pc1_max) / 2


def _apply_lateral_filter(
    report: _APNodePairReport,
    valid_idx: np.ndarray,
    lateral_thresh: float,
) -> np.ndarray | None:
    """Step 1: Filter keypoints by normalized lateral offset.

    Returns sorted candidate node indices, or None on failure.
    """
    d_valid = report.lateral_offsets[valid_idx]
    d_min = float(np.min(d_valid))
    d_max = float(np.max(d_valid))
    report.lateral_offset_min = d_min
    report.lateral_offset_max = d_max

    if d_max > d_min:
        d_norm = (d_valid - d_min) / (d_max - d_min)
        report.lateral_offsets_norm[valid_idx] = d_norm
        keep_mask = d_norm <= lateral_thresh
    else:
        report.lateral_offsets_norm[valid_idx] = np.zeros(len(d_valid))
        keep_mask = np.ones(len(d_valid), dtype=bool)

    candidate_idx = np.where(keep_mask)[0]
    C = valid_idx[candidate_idx]
    sorted_order = np.argsort(d_valid[candidate_idx])
    C = C[sorted_order]
    report.sorted_candidate_nodes = C.copy()

    if len(C) < 2:
        report.failure_step = "Step 1: lateral alignment filter"
        report.failure_reason = (
            "Fewer than 2 candidates remained after filtering."
        )
        return None
    return C


def _find_opposite_side_pairs(
    report: _APNodePairReport,
    C: np.ndarray,
    from_node: int,
    to_node: int,
    valid_shape_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Step 2: Find candidate pairs on opposite sides of the AP midpoint.

    Returns (pairs, seps) arrays, or None on failure.
    """
    m = report.midpoint_pc1
    report.input_pair_in_candidates = (from_node in C) and (to_node in C)

    pairs_list: list[list[int]] = []
    seps_list: list[float] = []
    for ii in range(len(C)):
        for jj in range(ii + 1, len(C)):
            i, j = C[ii], C[jj]
            if (report.pc1_coords[i] - m) * (report.pc1_coords[j] - m) < 0:
                pairs_list.append([i, j])
                seps_list.append(
                    abs(report.ap_coords[i] - report.ap_coords[j])
                )

    pairs = (
        np.array(pairs_list, dtype=int)
        if pairs_list
        else np.zeros((0, 2), dtype=int)
    )
    seps = np.array(seps_list) if seps_list else np.array([])
    report.valid_pairs = pairs
    report.valid_pairs_internode_dist = seps

    if valid_shape_rows[from_node] and valid_shape_rows[to_node]:
        report.input_pair_opposite_sides = (
            (report.pc1_coords[from_node] - m)
            * (report.pc1_coords[to_node] - m)
        ) < 0
        report.input_pair_separation_abs = abs(
            report.ap_coords[from_node] - report.ap_coords[to_node]
        )

    if len(pairs) == 0:
        report.failure_step = "Step 2: opposite-sides constraint"
        report.failure_reason = (
            "No candidate pair lies on opposite sides of the midpoint."
        )
        return None
    return pairs, seps


def _order_pair_by_ap(
    pair: np.ndarray,
    ap_coords: np.ndarray,
) -> np.ndarray:
    """Order a node pair so element 0 is posterior (lower AP coord).

    This ensures that suggested pairs always encode the
    posterior→anterior direction, matching the convention used by
    ``body_axis_keypoints=(from_node, to_node)`` where from_node is
    posterior and to_node is anterior.

    Parameters
    ----------
    pair : np.ndarray
        Two-element array of node indices.
    ap_coords : np.ndarray
        AP coordinates for all keypoints (anterior_sign already applied).

    Returns
    -------
    np.ndarray
        The same two indices, ordered so that
        ``ap_coords[result[0]] <= ap_coords[result[1]]``.

    """
    i, j = pair
    if ap_coords[i] <= ap_coords[j]:
        return np.array([i, j], dtype=int)
    return np.array([j, i], dtype=int)


def _classify_distal_proximal(
    report: _APNodePairReport,
    pairs: np.ndarray,
    seps: np.ndarray,
    valid_shape_rows: np.ndarray,
    edge_thresh: float,
) -> np.ndarray:
    """Step 3: Classify pairs as distal or proximal. Returns pair_is_distal."""
    m = report.midpoint_pc1
    midline_dist = np.abs(report.pc1_coords - m)
    d_max_midline = float(np.nanmax(midline_dist[valid_shape_rows]))
    report.midline_dist_max = d_max_midline

    if d_max_midline > 0:
        report.midline_dist_norm = midline_dist / d_max_midline
    else:
        report.midline_dist_norm = np.zeros(len(report.pc1_coords))

    pair_is_distal = np.zeros(len(pairs), dtype=bool)
    for k in range(len(pairs)):
        i, j = pairs[k]
        pair_is_distal[k] = (
            min(report.midline_dist_norm[i], report.midline_dist_norm[j])
            >= edge_thresh
        )

    report.distal_pairs = pairs[pair_is_distal]
    report.proximal_pairs = pairs[~pair_is_distal]

    if len(seps) > 0:
        idx_max = int(np.argmax(seps))
        report.max_separation_nodes = _order_pair_by_ap(
            pairs[idx_max], report.ap_coords
        )
        report.max_separation = seps[idx_max]

    if np.any(pair_is_distal):
        distal_seps = seps[pair_is_distal]
        distal_pairs_only = pairs[pair_is_distal]
        idx_max_distal = int(np.argmax(distal_seps))
        report.max_separation_distal_nodes = _order_pair_by_ap(
            distal_pairs_only[idx_max_distal], report.ap_coords
        )
        report.max_separation_distal = distal_seps[idx_max_distal]

    return pair_is_distal


def _check_input_pair_in_valid(
    report: _APNodePairReport,
    pairs: np.ndarray,
    seps: np.ndarray,
    pair_is_distal: np.ndarray,
    from_node: int,
    to_node: int,
) -> tuple[bool, int]:
    """Check whether input pair is among valid pairs. Returns (found, idx)."""
    input_pair_sorted = tuple(sorted([from_node, to_node]))
    input_in_valid = False
    input_idx = -1

    for k in range(len(pairs)):
        if tuple(sorted(pairs[k])) == input_pair_sorted:
            input_in_valid = True
            input_idx = k
            break

    if input_in_valid:
        report.input_pair_is_distal = pair_is_distal[input_idx]
        rank_order = np.argsort(seps)[::-1]
        report.input_pair_rank = (
            int(np.where(rank_order == input_idx)[0][0]) + 1
        )
    return input_in_valid, input_idx


def _evaluate_ap_node_pair(
    avg_skeleton: np.ndarray,
    PC1: np.ndarray,
    anterior_sign: int,
    valid_shape_rows: np.ndarray,
    from_node: int,
    to_node: int,
    config: _ValidateAPConfig,
) -> _APNodePairReport:
    """Evaluate an AP node pair through the 3-step filter cascade.

    Parameters
    ----------
    avg_skeleton : np.ndarray
        Average centered skeleton of shape (n_keypoints, 2).
    PC1 : np.ndarray
        First principal component vector of shape (2,).
    anterior_sign : int
        Inferred anterior direction (+1 or -1 relative to PC1).
    valid_shape_rows : np.ndarray
        Boolean array indicating valid (non-NaN) keypoints.
    from_node : int
        Index of the input from_node (body_axis_keypoints origin,
        claimed posterior). 0-indexed.
    to_node : int
        Index of the input to_node (body_axis_keypoints target,
        claimed anterior). 0-indexed.
    config : _ValidateAPConfig
        Configuration with ``lateral_thresh`` and ``edge_thresh``.

    Returns
    -------
    _APNodePairReport
        Complete evaluation report.

    """
    n_keypoints = len(avg_skeleton)
    report = _APNodePairReport()
    report.pc1_coords = np.full(n_keypoints, np.nan)
    report.ap_coords = np.full(n_keypoints, np.nan)
    report.lateral_offsets = np.full(n_keypoints, np.nan)
    report.lateral_offsets_norm = np.full(n_keypoints, np.nan)
    report.midline_dist_norm = np.full(n_keypoints, np.nan)

    for node, label in [(from_node, "from_node"), (to_node, "to_node")]:
        if node < 0 or node >= n_keypoints:
            report.failure_step = "Input validation"
            report.failure_reason = (
                f"{label} must be a valid index in 0..{n_keypoints - 1}."
            )
            return report

    valid_idx = np.where(valid_shape_rows)[0]
    if len(valid_idx) < 2:
        report.failure_step = "Step 1: lateral alignment filter"
        report.failure_reason = "Fewer than 2 valid nodes are available."
        return report

    _compute_node_projections(
        report,
        avg_skeleton,
        PC1,
        anterior_sign,
        valid_shape_rows,
        from_node,
        to_node,
    )

    C = _apply_lateral_filter(report, valid_idx, config.lateral_thresh)
    if C is None:
        return report

    step2 = _find_opposite_side_pairs(
        report,
        C,
        from_node,
        to_node,
        valid_shape_rows,
    )
    if step2 is None:
        return report
    pairs, seps = step2

    pair_is_distal = _classify_distal_proximal(
        report,
        pairs,
        seps,
        valid_shape_rows,
        config.edge_thresh,
    )

    input_in_valid, input_idx = _check_input_pair_in_valid(
        report,
        pairs,
        seps,
        pair_is_distal,
        from_node,
        to_node,
    )

    report = _assign_scenario(
        report, pairs, seps, pair_is_distal, input_in_valid, input_idx
    )
    report.success = True
    return report


def _assign_single_pair_scenario(
    report: _APNodePairReport,
    pairs: np.ndarray,
    pair_is_distal: np.ndarray,
    input_in_valid: bool,
) -> _APNodePairReport:
    """Assign scenario when exactly one valid pair exists (scenarios 1-4)."""
    if input_in_valid:
        if pair_is_distal[0]:
            report.scenario = 1
            report.outcome = "accept"
        else:
            report.scenario = 2
            report.outcome = "warn"
            report.warning_message = "Input pair has proximal node(s)."
    elif pair_is_distal[0]:
        report.scenario = 3
        report.outcome = "warn"
        report.warning_message = (
            f"Input invalid. Suggest pair [{pairs[0, 0]}, {pairs[0, 1]}]."
        )
    else:
        report.scenario = 4
        report.outcome = "warn"
        report.warning_message = (
            f"Input invalid. Only option "
            f"[{pairs[0, 0]}, {pairs[0, 1]}] has proximal node(s)."
        )
    return report


def _assign_multi_input_distal_scenario(
    report: _APNodePairReport,
    pairs: np.ndarray,
    input_idx: int,
) -> _APNodePairReport:
    """Assign scenario for distal input in multi-pair case (5, 6, 7)."""
    input_pair_sorted = tuple(
        sorted([pairs[input_idx, 0], pairs[input_idx, 1]])
    )
    max_distal_sorted = (
        tuple(sorted(report.max_separation_distal_nodes))
        if len(report.max_separation_distal_nodes) > 0
        else ()
    )

    if report.input_pair_rank == 1:
        report.scenario = 5
        report.outcome = "accept"
    elif input_pair_sorted == max_distal_sorted:
        report.scenario = 7
        report.outcome = "accept"
    else:
        report.scenario = 6
        report.outcome = "warn"
        d = report.max_separation_distal_nodes
        report.warning_message = (
            f"Distal pair with greater separation exists: [{d[0]}, {d[1]}]."
        )
    return report


def _assign_multi_input_proximal_scenario(
    report: _APNodePairReport,
    pair_is_distal: np.ndarray,
) -> _APNodePairReport:
    """Assign scenario for proximal input in multi-pair case (8-11)."""
    has_distal = np.any(pair_is_distal)
    is_max_sep = report.input_pair_rank == 1

    if is_max_sep and has_distal:
        report.scenario = 8
        d = report.max_separation_distal_nodes
        report.warning_message = (
            f"Input has proximal node(s). "
            f"Distal alternative: [{d[0]}, {d[1]}]."
        )
    elif is_max_sep:
        report.scenario = 9
        report.warning_message = (
            "Input has proximal node(s). All pairs have proximal node(s)."
        )
    elif has_distal:
        report.scenario = 10
        d = report.max_separation_distal_nodes
        report.warning_message = (
            f"Input has proximal node(s). "
            f"Distal pair with greater separation: [{d[0]}, {d[1]}]."
        )
    else:
        report.scenario = 11
        report.warning_message = (
            "Input has proximal node(s). All pairs have proximal node(s)."
        )

    report.outcome = "warn"
    return report


def _assign_multi_input_invalid_scenario(
    report: _APNodePairReport,
    pair_is_distal: np.ndarray,
) -> _APNodePairReport:
    """Assign scenario when input not in valid pairs (12-13)."""
    has_distal = np.any(pair_is_distal)
    report.outcome = "warn"

    if has_distal:
        report.scenario = 12
        d = report.max_separation_distal_nodes
        report.warning_message = (
            f"Input invalid. Suggest max separation distal pair: "
            f"[{d[0]}, {d[1]}]."
        )
    else:
        report.scenario = 13
        m = report.max_separation_nodes
        report.warning_message = (
            f"Input invalid. All pairs have proximal node(s). "
            f"Max separation: [{m[0]}, {m[1]}]."
        )
    return report


def _assign_scenario(
    report: _APNodePairReport,
    pairs: np.ndarray,
    seps: np.ndarray,
    pair_is_distal: np.ndarray,
    input_in_valid: bool,
    input_idx: int,
) -> _APNodePairReport:
    """Assign one of 13 mutually exclusive scenarios.

    Parameters
    ----------
    report : _APNodePairReport
        The report to update with scenario information.
    pairs : np.ndarray
        Valid pairs array of shape (n_pairs, 2).
    seps : np.ndarray
        Internode separations for each pair.
    pair_is_distal : np.ndarray
        Boolean array indicating distal pairs.
    input_in_valid : bool
        Whether input pair is among valid pairs.
    input_idx : int
        Index of input pair in valid pairs (-1 if not present).

    Returns
    -------
    _APNodePairReport
        Updated report with scenario, outcome, and warning_message.

    """
    if len(pairs) == 1:
        return _assign_single_pair_scenario(
            report,
            pairs,
            pair_is_distal,
            input_in_valid,
        )

    if not input_in_valid:
        return _assign_multi_input_invalid_scenario(report, pair_is_distal)

    if report.input_pair_is_distal:
        return _assign_multi_input_distal_scenario(
            report,
            pairs,
            input_idx,
        )

    return _assign_multi_input_proximal_scenario(report, pair_is_distal)


# ── _validate_ap helper functions ────────────────────────────────────────


def _resolve_node_index(node: Hashable, names: list) -> int:
    """Resolve a node identifier to an integer index."""
    if isinstance(node, str):
        if node in names:
            return names.index(node)
        raise ValueError(f"Keypoint '{node}' not found in {names}.")
    if isinstance(node, int):
        return node
    return int(node)  # type: ignore[call-overload]


def _prepare_validation_inputs(
    data: xr.DataArray,
    from_node: Hashable,
    to_node: Hashable,
) -> tuple[np.ndarray, int, int, str, str, list[str], int]:
    """Validate inputs and extract numpy arrays for AP validation.

    Returns
    -------
    tuple
        (keypoints, from_idx, to_idx, from_name, to_name,
         keypoint_names, num_frames)

    Raises
    ------
    TypeError
        If data is not an xarray.DataArray.
    ValueError
        If dimensions or indices are invalid.

    """
    _validate_type_data_array(data)

    required_dims = {"time", "space", "keypoints"}
    if not required_dims.issubset(set(data.dims)):
        raise ValueError(
            f"data must have dimensions {required_dims}, "
            f"but has {set(data.dims)}."
        )

    if "individuals" in data.dims:
        if data.sizes["individuals"] != 1:
            raise ValueError(
                "data must be for a single individual. "
                "Use data.sel(individuals='name') to select one."
            )
        data = data.squeeze("individuals", drop=True)

    if "keypoints" in data.coords:
        keypoint_names = list(data.coords["keypoints"].values)
    else:
        keypoint_names = [f"node_{i}" for i in range(data.sizes["keypoints"])]

    n_keypoints = data.sizes["keypoints"]
    from_idx = _resolve_node_index(from_node, keypoint_names)
    to_idx = _resolve_node_index(to_node, keypoint_names)

    if from_idx < 0 or from_idx >= n_keypoints:
        raise ValueError(
            f"from_node index {from_idx} out of range [0, {n_keypoints - 1}]."
        )
    if to_idx < 0 or to_idx >= n_keypoints:
        raise ValueError(
            f"to_node index {to_idx} out of range [0, {n_keypoints - 1}]."
        )

    data_xy = data.sel(space=["x", "y"])
    keypoints = data_xy.transpose("time", "keypoints", "space").values

    from_name = keypoint_names[from_idx]
    to_name = keypoint_names[to_idx]
    num_frames = keypoints.shape[0]

    return (
        keypoints,
        from_idx,
        to_idx,
        from_name,
        to_name,
        keypoint_names,
        num_frames,
    )


def _run_motion_segmentation(
    keypoints: np.ndarray,
    num_frames: int,
    config: _ValidateAPConfig,
    log_info,
    log_warning,
) -> dict | None:
    """Run tiered validity through segment detection.

    Returns a dict with tier1_valid, tier2_valid, bbox_centroids,
    segments, or None on failure (error logged).
    """
    tier1_valid, tier2_valid, _frac = _compute_tiered_validity(
        keypoints, config.min_valid_frac
    )
    num_tier1 = int(np.sum(tier1_valid))
    num_tier2 = int(np.sum(tier2_valid))

    log_info("────────────────────────────────────────────────────────────")
    log_info("Tiered Validity Report")
    log_info("────────────────────────────────────────────────────────────")
    log_info(
        "Tier 1 (>= %.0f%% keypoints): %d / %d frames (%.2f%%)",
        config.min_valid_frac * 100,
        num_tier1,
        num_frames,
        100 * num_tier1 / num_frames,
    )
    log_info(
        "Tier 2 (100%% keypoints):     %d / %d frames (%.2f%%)",
        num_tier2,
        num_frames,
        100 * num_tier2 / num_frames,
    )

    if num_tier1 < 2:
        logger.error("Not enough tier-1 valid frames.")
        return None

    bbox_centroids, _arith, centroid_disc = _compute_bbox_centroid(
        keypoints, tier1_valid
    )
    valid_disc = centroid_disc[tier1_valid & ~np.isnan(centroid_disc)]
    if len(valid_disc) > 0:
        log_info("")
        log_info(
            "────────────────────────────────────────────────────────────"
        )
        log_info("Centroid Discrepancy Diagnostic")
        log_info(
            "────────────────────────────────────────────────────────────"
        )
        log_info("BBox vs arithmetic centroid (normalized by bbox diagonal):")
        log_info(
            "  Median: %.4f | Mean: %.4f | Max: %.4f",
            np.median(valid_disc),
            np.mean(valid_disc),
            np.max(valid_disc),
        )
        if np.median(valid_disc) > 0.05:
            log_warning(
                "Median discrepancy > 5%% - annotation density "
                "is likely asymmetric."
            )

    segments = _detect_motion_segments(
        bbox_centroids, tier1_valid, config, log_info, log_warning
    )
    if segments is None:
        return None

    return {
        "tier1_valid": tier1_valid,
        "tier2_valid": tier2_valid,
        "bbox_centroids": bbox_centroids,
        "segments": segments,
    }


def _detect_motion_segments(
    bbox_centroids: np.ndarray,
    tier1_valid: np.ndarray,
    config: _ValidateAPConfig,
    log_info,
    log_warning,
) -> np.ndarray | None:
    """Detect high-motion segments from centroid velocities.

    Returns merged segments array, or None on failure.
    """
    velocities, speeds = _compute_frame_velocities(bbox_centroids, tier1_valid)
    num_speed = len(speeds)

    if num_speed < config.window_len:
        logger.error(
            "window_len=%d exceeds available speed samples=%d.",
            config.window_len,
            num_speed,
        )
        return None

    window_starts, window_medians, window_all_valid = (
        _compute_sliding_window_medians(
            speeds, config.window_len, config.stride
        )
    )
    num_valid_windows = int(np.sum(window_all_valid))
    if num_valid_windows == 0:
        logger.error("No fully valid sliding windows found.")
        return None

    high_motion = _detect_high_motion_windows(
        window_medians, window_all_valid, config.pct_thresh
    )
    num_high_motion = int(np.sum(high_motion))

    log_info("")
    log_info("────────────────────────────────────────────────────────────")
    log_info("High-Motion Window Detection")
    log_info("────────────────────────────────────────────────────────────")
    log_info(
        "Sliding windows (len=%d, stride=%d): "
        "%d total, %d fully valid (NaN-free), "
        "%d high-motion (median speed >= %dth percentile)",
        config.window_len,
        config.stride,
        len(window_starts),
        num_valid_windows,
        num_high_motion,
        int(config.pct_thresh),
    )

    if num_high_motion == 0:
        logger.error("No high-motion windows found.")
        return None

    run_starts, run_ends, _run_lengths = _detect_runs(
        high_motion, config.min_run_len
    )
    if len(run_starts) == 0:
        logger.error("No runs met min_run_len=%d.", config.min_run_len)
        return None

    segments_raw = _convert_runs_to_segments(
        run_starts, run_ends, window_starts, config.window_len
    )
    segments = _merge_segments(segments_raw)

    log_info("Detected %d merged high-motion segment(s):", len(segments))
    for i, (start, end) in enumerate(segments):
        log_info("  Segment %d: frames %d - %d", i + 1, start, end)

    return segments


def _select_tier2_frames(
    segments: np.ndarray,
    tier2_valid: np.ndarray,
    num_frames: int,
    log_info,
    log_warning,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Filter segment frames to tier-2 valid only.

    Returns (selected_frames, selected_seg_id, num_selected) or None.
    """
    selected_frames, selected_seg_id = _filter_segments_tier2(
        segments, tier2_valid
    )

    num_tier1_in_segs = sum(
        np.sum(
            (np.arange(num_frames) >= s[0]) & (np.arange(num_frames) <= s[1])
        )
        for s in segments
    )
    num_selected = len(selected_frames)

    log_info("")
    log_info("────────────────────────────────────────────────────────────")
    log_info("Tier-2 Filtering on High-Motion Segments")
    log_info("────────────────────────────────────────────────────────────")
    log_info(
        "Frames in high-motion segments (any tier): %d", num_tier1_in_segs
    )
    log_info(
        "Tier-2 valid frames retained (all keypoints present): "
        "%d (%.1f%% of segment frames)",
        num_selected,
        100 * num_selected / max(num_tier1_in_segs, 1),
    )

    retention = num_selected / max(num_tier1_in_segs, 1)
    if retention < 0.3:
        log_warning(
            "Tier 2 discards > 70%% of segment frames - "
            "body model may be unrepresentative."
        )

    if num_selected < 2:
        logger.error("Not enough tier-2 valid frames in selected segments.")
        return None

    return selected_frames, selected_seg_id, num_selected


def _run_clustering_and_pca(
    centered_skeletons: np.ndarray,
    frame_sel: _FrameSelection,
    config: _ValidateAPConfig,
    log_info,
    log_warning,
) -> dict | None:
    """Run postural analysis, clustering, and per-cluster PCA.

    Returns dict with primary_result, cluster_results,
    num_clusters, primary_cluster, or None on failure.
    """
    rmsd_matrix = _compute_pairwise_rmsd(centered_skeletons)
    var_ratio, within_rmsds, between_rmsds, var_ratio_override = (
        _compute_postural_variance_ratio(rmsd_matrix, frame_sel.seg_ids)
    )

    rmsd_stats = {
        "within": within_rmsds,
        "between": between_rmsds,
        "var_ratio": var_ratio,
        "override": var_ratio_override,
    }
    _log_postural_consistency(
        rmsd_stats,
        config,
        frame_sel.count,
        log_info,
        log_warning,
    )

    cluster_labels, num_clusters, primary_cluster = _decide_and_run_clustering(
        centered_skeletons,
        var_ratio,
        frame_sel.count,
        config,
        log_info,
    )

    cluster_results = []
    for c in range(num_clusters):
        cluster_mask = cluster_labels == c
        cr = _compute_cluster_pca_and_anterior(
            centered_skeletons,
            cluster_mask,
            frame_sel.frames,
            frame_sel.seg_ids,
            frame_sel.segments,
            frame_sel.bbox_centroids,
        )
        cluster_results.append(cr)

    pr = cluster_results[primary_cluster]
    if not pr["valid"]:
        logger.error("Primary cluster has invalid PCA result.")
        return None

    return {
        "primary_result": pr,
        "cluster_results": cluster_results,
        "num_clusters": num_clusters,
        "primary_cluster": primary_cluster,
    }


def _log_postural_consistency(
    rmsd_stats,
    config,
    num_selected,
    log_info,
    log_warning,
):
    """Log postural consistency check results."""
    within_rmsds = rmsd_stats["within"]
    between_rmsds = rmsd_stats["between"]
    var_ratio = rmsd_stats["var_ratio"]
    var_ratio_override = rmsd_stats["override"]

    log_info("")
    log_info("────────────────────────────────────────────────────────────")
    log_info("Postural Consistency Check")
    log_info("────────────────────────────────────────────────────────────")

    if len(within_rmsds) > 0:
        log_info(
            "Within-segment RMSD:  mean=%.4f, std=%.4f (n=%d pairs)",
            np.mean(within_rmsds),
            np.std(within_rmsds),
            len(within_rmsds),
        )
    else:
        log_info("Within-segment RMSD:  N/A (no within-segment pairs)")

    if len(between_rmsds) > 0:
        log_info(
            "Between-segment RMSD: mean=%.4f, std=%.4f (n=%d pairs)",
            np.mean(between_rmsds),
            np.std(between_rmsds),
            len(between_rmsds),
        )
        log_info(
            "Variance ratio (between/within): %.2f (threshold=%.2f)",
            var_ratio,
            config.postural_var_ratio_thresh,
        )
        if var_ratio_override:
            log_info(
                "  (Conservative override to zero: within-segment variance "
                "is zero or no within-segment pairs)"
            )
    else:
        log_info("Between-segment RMSD: N/A (single segment)")
        log_info("Variance ratio: N/A")

    do_clustering = (
        var_ratio > config.postural_var_ratio_thresh and num_selected >= 6
    )
    if do_clustering:
        log_info("  -> Variance ratio exceeds threshold. Running clustering.")
    elif var_ratio > config.postural_var_ratio_thresh and num_selected < 6:
        log_info(
            "  -> Variance ratio exceeds threshold but too few frames (%d) "
            "for clustering.",
            num_selected,
        )
    else:
        log_info("  -> Postural consistency acceptable. Using global average.")


def _decide_and_run_clustering(
    centered_skeletons,
    var_ratio,
    num_selected,
    config,
    log_info,
):
    """Decide whether to cluster; run k-medoids if triggered."""
    do_clustering = (
        var_ratio > config.postural_var_ratio_thresh and num_selected >= 6
    )

    if not do_clustering:
        return np.zeros(num_selected, dtype=int), 1, 0

    (
        cluster_labels,
        num_clusters,
        primary_cluster,
        best_silhouette,
        silhouette_scores,
    ) = _perform_postural_clustering(centered_skeletons, config.max_clusters)

    for k, sil in silhouette_scores:
        if np.isnan(sil):
            log_info("  k=%d: clustering failed.", k)
        else:
            log_info("  k=%d: mean silhouette = %.4f", k, sil)

    if num_clusters > 1:
        cluster_counts = np.bincount(cluster_labels, minlength=num_clusters)
        log_info(
            "  Selected k=%d clusters (silhouette=%.4f). "
            "Primary cluster=%d (%d frames)",
            num_clusters,
            best_silhouette,
            primary_cluster + 1,
            cluster_counts[primary_cluster],
        )
    else:
        log_info(
            "  Clustering did not improve separation (best_sil=%.4f). "
            "Using global average.",
            best_silhouette,
        )

    return cluster_labels, num_clusters, primary_cluster


def _log_anterior_report(
    pr,
    cluster_results,
    num_clusters,
    primary_cluster,
    config,
    log_info,
    log_warning,
):
    """Log anterior direction detection and cluster agreement."""
    log_info("")
    log_info("────────────────────────────────────────────────────────────")
    log_info("Anterior Direction Inference (Velocity Voting)")
    log_info("────────────────────────────────────────────────────────────")
    log_info(
        "Centroid velocity projections onto PC1: "
        "%d positive (+PC1), %d negative (−PC1)",
        pr["num_positive"],
        pr["num_negative"],
    )

    vote_margin_str = f"Vote margin M: {pr['vote_margin']:.4f}"
    if pr["vote_margin"] < config.confidence_floor:
        vote_margin_str += (
            f"  ** BELOW CONFIDENCE FLOOR ({config.confidence_floor:.2f}) "
            "— anterior assignment is unreliable **"
        )
        log_warning(
            "Vote margin M = %.4f is below confidence floor %.2f — "
            "anterior assignment is unreliable.",
            pr["vote_margin"],
            config.confidence_floor,
        )
    log_info(vote_margin_str)
    log_info(
        "Resultant length R: %.4f (0 = omnidirectional, 1 = unidirectional)",
        pr["resultant_length"],
    )
    log_info(
        "Inferred anterior direction: %sPC1 "
        "(strict majority; ties default to −PC1)",
        "+" if pr["anterior_sign"] > 0 else "−",
    )

    if num_clusters <= 1:
        return

    log_info("")
    log_info("────────────────────────────────────────────────────────────")
    log_info("Inter-Cluster Anterior Polarity Agreement")
    log_info("────────────────────────────────────────────────────────────")
    signs = [cr["anterior_sign"] for cr in cluster_results if cr["valid"]]
    if len(set(signs)) == 1:
        log_info(
            "All %d clusters AGREE on anterior polarity.",
            num_clusters,
        )
    else:
        log_info(
            "DISAGREEMENT: clusters assign different anterior polarities."
        )
        for c, cr in enumerate(cluster_results):
            if cr["valid"]:
                log_info(
                    "  Cluster %d (%d frames): anterior = %sPC1, "
                    "vote_margin M = %.4f, resultant_length R = %.4f",
                    c + 1,
                    cr["n_frames"],
                    "+" if cr["anterior_sign"] > 0 else "−",
                    cr["vote_margin"],
                    cr["resultant_length"],
                )
        log_info(
            "  Primary result from cluster %d (largest).",
            primary_cluster + 1,
        )


def _log_step1_report(pair_report, config, valid_nodes, log_info):
    """Log Step 1 lateral filter results."""
    num_valid = len(valid_nodes)
    num_candidates = len(pair_report.sorted_candidate_nodes)
    step1_loss = 1 - num_candidates / max(num_valid, 1)

    pass_strs = []
    fail_strs = []
    for node_i in valid_nodes:
        lat_norm = pair_report.lateral_offsets_norm[node_i]
        if lat_norm <= config.lateral_thresh:
            pass_strs.append(f"{node_i}({lat_norm:.2f})")
        else:
            fail_strs.append(f"{node_i}({lat_norm:.2f})")

    log_info("")
    log_info(
        "Step 1 — Lateral Alignment Filter (lateral_thresh=%.2f): "
        "%d of %d valid nodes pass [loss=%.0f%%]",
        config.lateral_thresh,
        num_candidates,
        num_valid,
        100 * step1_loss,
    )
    log_info(
        "  Scale: 0.00 = nearest to body axis, 1.00 = farthest from body axis"
    )
    if pass_strs:
        log_info("  PASS: %s", ", ".join(pass_strs))
    if fail_strs:
        log_info("  FAIL: %s", ", ".join(fail_strs))


def _log_step2_report(pair_report, config, log_info):
    """Log Step 2 opposite-sides results."""
    num_candidates = len(pair_report.sorted_candidate_nodes)
    num_possible_pairs = num_candidates * (num_candidates - 1) // 2
    num_valid_pairs = len(pair_report.valid_pairs)
    step2_loss = 1 - num_valid_pairs / max(num_possible_pairs, 1)
    m = pair_report.midpoint_pc1

    plus_strs = []
    minus_strs = []
    for node_i in pair_report.sorted_candidate_nodes:
        pc1_rel = pair_report.pc1_coords[node_i] - m
        if pc1_rel > 0:
            plus_strs.append(f"{node_i}({pc1_rel:+.1f})")
        else:
            minus_strs.append(f"{node_i}({pc1_rel:+.1f})")

    log_info("")
    log_info(
        "Step 2 — Opposite-Sides Constraint (AP midpoint=%.2f): "
        "%d of %d candidate pairs on opposite sides [loss=%.0f%%]",
        m,
        num_valid_pairs,
        num_possible_pairs,
        100 * step2_loss,
    )
    if plus_strs:
        log_info("  + side (anterior of midpoint): %s", ", ".join(plus_strs))
    if minus_strs:
        log_info("  - side (posterior of midpoint): %s", ", ".join(minus_strs))


def _log_step3_report(pair_report, config, log_info):
    """Log Step 3 distal/proximal classification results."""
    num_distal = len(pair_report.distal_pairs)
    num_proximal = len(pair_report.proximal_pairs)
    num_valid_pairs = len(pair_report.valid_pairs)
    step3_distal_frac = num_distal / max(num_valid_pairs, 1)

    log_info("")
    log_info(
        "Step 3 — Distal/Proximal Classification (edge_thresh=%.2f): "
        "%d distal, %d proximal [distal fraction=%.0f%%]",
        config.edge_thresh,
        num_distal,
        num_proximal,
        100 * step3_distal_frac,
    )

    for idx in range(num_valid_pairs):
        node_i, node_j = pair_report.valid_pairs[idx]
        d_i = pair_report.midline_dist_norm[node_i]
        d_j = pair_report.midline_dist_norm[node_j]
        min_d = min(d_i, d_j)
        sep = pair_report.valid_pairs_internode_dist[idx]
        status = "DISTAL" if min_d >= config.edge_thresh else "PROXIMAL"
        log_info(
            "  [%d,%d]: min_d=%.2f, sep=%.2f [%s]",
            node_i,
            node_j,
            min_d,
            sep,
            status,
        )


def _log_loss_summary(
    step1_loss, step2_loss, step3_frac, step1_failed, step2_failed, log_info
):
    """Log cumulative filtering loss summary."""
    log_info("")
    log_info("────────────────────────────────────────────────────────────")
    log_info("Filtering Loss Summary")
    log_info("────────────────────────────────────────────────────────────")
    log_info(
        "Step 1 (Lateral Filter): %.0f%% of valid nodes eliminated",
        100 * step1_loss,
    )
    if not step1_failed:
        log_info(
            "Step 2 (Opposite-Sides): %.0f%% of candidate pairs eliminated",
            100 * step2_loss,
        )
    if not step1_failed and not step2_failed:
        log_info(
            "Step 3 (Distal/Proximal): %.0f%% of surviving pairs are distal",
            100 * step3_frac,
        )


def _log_order_check(
    pair_report, from_idx, to_idx, from_name, to_name, log_info
):
    """Log AP ordering check for the input pair."""
    log_info("")
    log_info("────────────────────────────────────────────────────────────")
    log_info("Order Check: is from_node posterior to to_node?")
    log_info("────────────────────────────────────────────────────────────")
    ap_from = pair_report.ap_coords[from_idx]
    ap_to = pair_report.ap_coords[to_idx]
    if np.isnan(ap_from) or np.isnan(ap_to):
        log_info("Order check: cannot evaluate (invalid node coordinates)")
        return

    log_info(
        "AP coords: from_node %s[%d]=%.2f, to_node %s[%d]=%.2f",
        from_name,
        from_idx,
        ap_from,
        to_name,
        to_idx,
        ap_to,
    )
    neon_green = "\033[38;5;46m"
    crimson = "\033[38;5;9m"
    reset = "\033[0m"
    if pair_report.input_pair_order_matches_inference:
        log_info(
            f"{neon_green}[%d, %d]: CONSISTENT — inference agrees that "
            f"from_node is posterior (lower AP coord), "
            f"to_node is anterior{reset}",
            from_idx,
            to_idx,
        )
    else:
        log_info(
            f"{crimson}[%d, %d]: INCONSISTENT — inference suggests "
            f"from_node is anterior (higher AP coord), "
            f"to_node is posterior{reset}",
            from_idx,
            to_idx,
        )
        log_info(
            "  -> Inferred posterior→anterior order would be [%d, %d]",
            to_idx,
            from_idx,
        )


def _log_input_node_status(
    pair_report,
    config,
    from_idx,
    to_idx,
    log_info,
):
    """Log whether each input node passed the lateral filter."""
    lat_from = pair_report.lateral_offsets_norm[from_idx]
    lat_to = pair_report.lateral_offsets_norm[to_idx]
    from_pass = not np.isnan(lat_from) and lat_from <= config.lateral_thresh
    to_pass = not np.isnan(lat_to) and lat_to <= config.lateral_thresh

    if from_pass and to_pass:
        return

    fail_nodes = []
    if not from_pass:
        fail_nodes.append(f"{from_idx}({lat_from:.2f})")
    if not to_pass:
        fail_nodes.append(f"{to_idx}({lat_to:.2f})")
    log_info(
        "  -> Input node(s) FAILED lateral filter: %s",
        ", ".join(fail_nodes),
    )


def _log_step2_step3_details(
    pair_report,
    config,
    from_idx,
    to_idx,
    num_candidates,
    log_info,
):
    """Log Step 2 and Step 3 results when Step 1 succeeded.

    Returns (step2_loss, step3_frac, step2_failed).
    """
    _log_step2_report(pair_report, config, log_info)

    if (
        pair_report.input_pair_in_candidates
        and not pair_report.input_pair_opposite_sides
    ):
        log_info("  -> Input nodes on SAME side of AP midpoint")

    num_possible = num_candidates * (num_candidates - 1) // 2
    step2_loss = 1 - len(pair_report.valid_pairs) / max(num_possible, 1)
    step2_failed = pair_report.failure_step.startswith("Step 2")

    if step2_failed:
        log_info("")
        log_info("Step 3: not evaluated (Step 2 failed)")
        return step2_loss, 0.0, True

    step3_frac = _log_step3_with_proximal_check(
        pair_report,
        config,
        from_idx,
        to_idx,
        log_info,
    )
    return step2_loss, step3_frac, False


def _log_step3_with_proximal_check(
    pair_report,
    config,
    from_idx,
    to_idx,
    log_info,
):
    """Log Step 3 results and check input pair proximal status.

    Returns step3_frac.
    """
    _log_step3_report(pair_report, config, log_info)
    num_distal = len(pair_report.distal_pairs)
    num_valid_pairs = len(pair_report.valid_pairs)
    step3_frac = num_distal / max(num_valid_pairs, 1)

    is_candidate = pair_report.input_pair_in_candidates
    is_opposite = pair_report.input_pair_opposite_sides
    is_proximal = not pair_report.input_pair_is_distal
    if is_candidate and is_opposite and is_proximal:
        d_from = pair_report.midline_dist_norm[from_idx]
        d_to = pair_report.midline_dist_norm[to_idx]
        log_info(
            "  -> Input pair is PROXIMAL (min_d=%.2f < %.2f)",
            min(d_from, d_to),
            config.edge_thresh,
        )

    return step3_frac


def _log_pair_evaluation(
    pair_report,
    config,
    from_idx,
    to_idx,
    from_name,
    to_name,
    log_info,
):
    """Log the complete AP node pair evaluation report."""
    log_info("")
    log_info("────────────────────────────────────────────────────────────")
    log_info("AP Node-Pair Filter Cascade (3-Step Evaluation)")
    log_info("────────────────────────────────────────────────────────────")
    log_info(
        "Input pair: [%d, %d] (%s → %s, claimed posterior → anterior)",
        from_idx,
        to_idx,
        from_name,
        to_name,
    )

    step1_failed = pair_report.failure_step.startswith("Step 1")

    valid_nodes = np.where(~np.isnan(pair_report.lateral_offsets_norm))[0]
    num_candidates = len(pair_report.sorted_candidate_nodes)
    step1_loss = 1 - num_candidates / max(len(valid_nodes), 1)

    _log_step1_report(pair_report, config, valid_nodes, log_info)
    _log_input_node_status(pair_report, config, from_idx, to_idx, log_info)

    step2_loss = 0.0
    step3_frac = 0.0
    step2_failed = False

    if step1_failed:
        log_info("")
        log_info("Step 2-3: not evaluated (Step 1 failed)")
    else:
        step2_loss, step3_frac, step2_failed = _log_step2_step3_details(
            pair_report,
            config,
            from_idx,
            to_idx,
            num_candidates,
            log_info,
        )

    _log_loss_summary(
        step1_loss,
        step2_loss,
        step3_frac,
        step1_failed,
        step2_failed,
        log_info,
    )
    _log_order_check(
        pair_report,
        from_idx,
        to_idx,
        from_name,
        to_name,
        log_info,
    )


# ── Main validation function ────────────────────────────────────────────


def _validate_ap(
    data: xr.DataArray,
    from_node: Hashable,
    to_node: Hashable,
    config: _ValidateAPConfig | None = None,
    verbose: bool = True,
) -> dict:
    """Validate an anterior-posterior keypoint pair using body-axis inference.

    This function implements a prior-free body-axis inference pipeline that:
    1. Identifies high-motion segments using tiered validity and sliding
       windows
    2. Optionally performs postural clustering via k-medoids
    3. Infers the anterior direction using velocity projection voting
    4. Evaluates the candidate AP keypoint pair through a 3-step filter
       cascade

    Parameters
    ----------
    data : xarray.DataArray
        Position data for a single individual.
    from_node : int or str
        Index or name of the posterior keypoint.
    to_node : int or str
        Index or name of the anterior keypoint.
    config : _ValidateAPConfig, optional
        Configuration parameters. If None, uses defaults.
    verbose : bool, default=True
        If True, log detailed validation output to console.

    Returns
    -------
    dict
        Validation results including success, anterior_sign,
        vote_margin, resultant_length, pair_report, etc.

    """
    if config is None:
        config = _ValidateAPConfig()

    log_lines: list[str] = []

    def _log_info(msg, *args):
        """Log an informational message."""
        line = msg % args if args else msg
        log_lines.append(line)
        if verbose:
            print(line)

    def _log_warning(msg, *args):
        """Log a warning message with ANSI coloring."""
        orange = "\033[38;5;214m"
        reset = "\033[0m"
        line = f"{orange}WARNING: {msg % args if args else msg}{reset}"
        log_lines.append(line)
        if verbose:
            print(line)

    # Prepare inputs
    (
        keypoints,
        from_idx,
        to_idx,
        from_name,
        to_name,
        _keypoint_names,
        num_frames,
    ) = _prepare_validation_inputs(data, from_node, to_node)

    n_keypoints = keypoints.shape[1]
    result: dict = {
        "success": False,
        "anterior_sign": 0,
        "vote_margin": 0.0,
        "resultant_length": 0.0,
        "num_selected_frames": 0,
        "num_clusters": 1,
        "primary_cluster": 0,
        "pair_report": _APNodePairReport(),
        "PC1": np.array([1.0, 0.0]),
        "PC2": np.array([0.0, 1.0]),
        "avg_skeleton": np.full((n_keypoints, 2), np.nan),
        "error_msg": "",
        "log_lines": log_lines,
    }

    # Motion segmentation
    seg = _run_motion_segmentation(
        keypoints,
        num_frames,
        config,
        _log_info,
        _log_warning,
    )
    if seg is None:
        result["error_msg"] = "Motion segmentation failed."
        return result

    # Tier-2 frame selection
    t2 = _select_tier2_frames(
        seg["segments"],
        seg["tier2_valid"],
        num_frames,
        _log_info,
        _log_warning,
    )
    if t2 is None:
        result["error_msg"] = "Not enough tier-2 valid frames."
        return result
    selected_frames, selected_seg_id, num_selected = t2
    result["num_selected_frames"] = num_selected

    # Build centered skeletons
    _selected_centroids, centered_skeletons = _build_centered_skeletons(
        keypoints, selected_frames
    )

    # Bundle frame selection data
    frame_sel = _FrameSelection(
        frames=selected_frames,
        seg_ids=selected_seg_id,
        segments=seg["segments"],
        bbox_centroids=seg["bbox_centroids"],
        count=num_selected,
    )

    # Postural clustering + PCA + anterior inference
    pca = _run_clustering_and_pca(
        centered_skeletons,
        frame_sel,
        config,
        _log_info,
        _log_warning,
    )
    if pca is None:
        result["error_msg"] = "Primary cluster PCA failed."
        return result

    pr = pca["primary_result"]
    result["anterior_sign"] = pr["anterior_sign"]
    result["vote_margin"] = pr["vote_margin"]
    result["resultant_length"] = pr["resultant_length"]
    result["circ_mean_dir"] = pr["circ_mean_dir"]
    result["vel_projs_pc1"] = pr["vel_projs_pc1"]
    result["PC1"] = pr["PC1"]
    result["PC2"] = pr["PC2"]
    result["avg_skeleton"] = pr["avg_skeleton"]
    result["num_clusters"] = pca["num_clusters"]
    result["primary_cluster"] = pca["primary_cluster"]

    # Log anterior inference
    _log_anterior_report(
        pr,
        pca["cluster_results"],
        pca["num_clusters"],
        pca["primary_cluster"],
        config,
        _log_info,
        _log_warning,
    )

    # AP node-pair evaluation
    pair_report = _evaluate_ap_node_pair(
        pr["avg_skeleton"],
        pr["PC1"],
        pr["anterior_sign"],
        pr["valid_shape_rows"],
        from_idx,
        to_idx,
        config,
    )
    result["pair_report"] = pair_report

    _log_pair_evaluation(
        pair_report,
        config,
        from_idx,
        to_idx,
        from_name,
        to_name,
        _log_info,
    )

    result["success"] = True
    return result
