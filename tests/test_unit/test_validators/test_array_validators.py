import re

import pytest

from movement.validators.arrays import validate_dims_coords


@pytest.mark.parametrize(
    "required_dims_coords",
    [
        ({"time": []}),
        ({"time": [0, 1]}),
        ({"space": ["x", "y"]}),
        ({"time": [], "space": []}),
        ({"time": [], "space": ["x", "y"]}),
    ],
)
def test_validate_dims_coords_on_valid_input(
    valid_poses_dataset_uniform_linear_motion,  # fixture from conftest.py
    required_dims_coords,
):
    """Test that valid inputs do not raise an error."""
    position_array = valid_poses_dataset_uniform_linear_motion["position"]
    validate_dims_coords(position_array, required_dims_coords)


@pytest.mark.parametrize(
    "required_dims_coords, expected_error_message",
    [
        (
            {"spacetime": []},
            "Input data must contain ['spacetime'] as dimensions.",
        ),
        (
            {"time": [0, 100], "space": ["x", "y"]},
            "Input data must contain [100] in the 'time' coordinates.",
        ),
        (
            {"space": ["x", "y", "z"]},
            "Input data must contain ['z'] in the 'space' coordinates.",
        ),
    ],
)
def test_validate_dims_coords_on_invalid_input(
    valid_poses_dataset_uniform_linear_motion,  # fixture from conftest.py
    required_dims_coords,
    expected_error_message,
):
    """Test that invalid inputs raise a ValueError with expected message."""
    position_array = valid_poses_dataset_uniform_linear_motion["position"]
    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        validate_dims_coords(position_array, required_dims_coords)
