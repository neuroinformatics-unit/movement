"""Module for computing trajectory complexity metrics."""

import xarray as xr

from movement.kinematics import compute_path_length
from movement.utils.logging import log_to_attrs
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


@log_to_attrs
def compute_straightness_index(
    data: xr.DataArray,
) -> xr.DataArray:
    """Compute the straightness index of a trajectory.

    The straightness index is defined as the ratio of the Euclidean distance
    between the start and end points of a trajectory to the total path
    length (D / L).
    Values range from 0 to 1, where 1 indicates a perfectly straight path.
    """
    validate_dims_coords(data, {"time": [], "space": []})

    start_point = data.isel(time=0)
    end_point = data.isel(time=-1)
    distance = compute_norm(end_point - start_point)

    path_length = compute_path_length(data)

    return distance / path_length
