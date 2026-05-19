"""Split trajectories based on NaN gaps."""

import numpy as np
import xarray as xr

from movement.validators.arrays import validate_dims_coords


def split_trajectories_by_gap(
    position: xr.DataArray,
    min_gap_size: int = 1,
) -> xr.DataArray:
    """Split trajectories into new individuals whenever there is a gap.

    A gap is defined as a consecutive sequence of NaN frames of length
    >= ``min_gap_size``. Each continuous segment of valid data is assigned
    a new unique individual ID.

    Parameters
    ----------
    position : xarray.DataArray
        The input position data with dimensions
        ``("time", "space", "keypoints", "individuals")``.
    min_gap_size : int, optional
        The minimum number of consecutive NaN frames required to trigger
        a trajectory split. Default is 1 (any NaN frame causes a split).

    Returns
    -------
    xarray.DataArray
        A new DataArray with the same data but with updated ``individuals``
        coordinate. Each continuous segment of valid data for an original
        individual becomes a new individual with a unique ID.

    Notes
    -----
    The naming convention for new individual IDs is:
    ``<original_id>_<segment_number:03d>``, e.g., "mouse0_000", "mouse0_001".

    A frame is considered valid if **any** feature (space × keypoints) is
    non-NaN for that individual.

    Examples
    --------
    Suppose we have position data with one individual and a gap at frames 2-3:

    >>> # Time:  0 1 2 3 4 5
    >>> # Valid: T T F F T T
    >>> result = split_trajectories_by_gap(position, min_gap_size=1)
    >>> # Result has two individuals: original_id_000 (frames 0-1)
    >>> #                             original_id_001 (frames 4-5)

    With ``min_gap_size=3``, the same data would not be split since the gap
    (length 2) is smaller than the threshold.

    """
    # Validate input dimensions
    validate_dims_coords(
        position,
        {
            "time": [],
            "space": [],
            "keypoints": [],
            "individuals": [],
        },
    )

    if min_gap_size < 1:
        raise ValueError("min_gap_size must be >= 1")

    # Get dimension sizes and coordinates
    time_coords = position.coords["time"].values
    space_coords = position.coords["space"].values
    keypoint_coords = position.coords["keypoints"].values
    individual_coords = position.coords["individuals"].values

    n_time = len(time_coords)
    n_space = len(space_coords)
    n_keypoints = len(keypoint_coords)

    # Collect all segments across all individuals
    all_segments_data = []
    all_segment_ids = []

    for indiv in individual_coords:
        # Extract data for this individual: shape (time, space, keypoints)
        indiv_data = position.sel(individuals=indiv).values

        # Flatten to (time, features) where features = space * keypoints
        flat_data = indiv_data.reshape(n_time, -1)

        # A frame is valid if ANY feature is non-NaN
        valid_frames = ~np.all(np.isnan(flat_data), axis=1)

        # Find segments separated by gaps >= min_gap_size
        segments = _find_segments(valid_frames, min_gap_size)

        # Create data arrays for each segment
        for seg_idx, (start, end) in enumerate(segments):
            # Create a full array of NaNs for this segment
            segment_data = np.full_like(indiv_data, np.nan)
            # Copy only the valid segment data
            segment_data[start:end, :, :] = indiv_data[start:end, :, :]

            # Generate unique ID: originalID_XXX
            new_id = f"{indiv}_{seg_idx:03d}"

            all_segments_data.append(segment_data)
            all_segment_ids.append(new_id)

    # Handle edge case: no valid segments found
    if not all_segments_data:
        # Return an empty DataArray with consistent structure
        empty_data = np.empty((n_time, n_space, n_keypoints, 0))
        return xr.DataArray(
            empty_data,
            dims=("time", "space", "keypoints", "individuals"),
            coords={
                "time": time_coords,
                "space": space_coords,
                "keypoints": keypoint_coords,
                "individuals": [],
            },
            attrs=position.attrs.copy(),
        )

    # Stack all segments along the individuals dimension
    # Each segment_data has shape (time, space, keypoints)
    stacked_data = np.stack(all_segments_data, axis=-1)

    # Create the new DataArray
    result = xr.DataArray(
        stacked_data,
        dims=("time", "space", "keypoints", "individuals"),
        coords={
            "time": time_coords,
            "space": space_coords,
            "keypoints": keypoint_coords,
            "individuals": all_segment_ids,
        },
        attrs=position.attrs.copy(),
    )

    return result


def _find_segments(
    valid_frames: np.ndarray,
    min_gap_size: int,
) -> list[tuple[int, int]]:
    """Find continuous segments in a boolean array.

    Parameters
    ----------
    valid_frames : np.ndarray
        1D boolean array where True indicates a valid frame.
    min_gap_size : int
        Minimum consecutive False values to be considered a gap.

    Returns
    -------
    list of tuple[int, int]
        List of (start, end) indices for each segment. The end index is
        exclusive (Python slice convention).

    """
    n_frames = len(valid_frames)

    if n_frames == 0:
        return []

    # Find runs of invalid (NaN) frames
    # We'll identify gaps and then determine segments between them

    # Find where valid changes from True to False or vice versa
    # Pad with False at both ends to handle edge cases
    padded = np.concatenate([[False], valid_frames, [False]])
    diff = np.diff(padded.astype(int))

    # Start of valid runs: where diff == 1
    starts = np.where(diff == 1)[0]
    # End of valid runs: where diff == -1
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        # No valid frames at all
        return []

    # Now we have valid segments defined by (starts[i], ends[i])
    # We need to merge segments that are separated by gaps < min_gap_size

    segments = []
    current_start = starts[0]
    current_end = ends[0]

    for i in range(1, len(starts)):
        gap_length = starts[i] - current_end
        if gap_length < min_gap_size:
            # Gap is too small, merge with current segment
            current_end = ends[i]
        else:
            # Gap is large enough, finalize current segment
            segments.append((current_start, current_end))
            current_start = starts[i]
            current_end = ends[i]

    # Don't forget the last segment
    segments.append((current_start, current_end))

    return segments
