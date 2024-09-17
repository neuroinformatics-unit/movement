import re
from contextlib import nullcontext as does_not_raise

import pytest

from movement.validators.arrays import validate_dims_coords


def expect_value_error_with_message(error_msg):
    """Expect a ValueError with the specified error message."""
    return pytest.raises(ValueError, match=re.escape(error_msg))


valid_cases = [
    ({"time": []}, does_not_raise()),
    ({"time": [0, 1]}, does_not_raise()),
    ({"space": ["x", "y"]}, does_not_raise()),
    ({"time": [], "space": []}, does_not_raise()),
    ({"time": [], "space": ["x", "y"]}, does_not_raise()),
]  # Valid cases (no error)

invalid_cases = [
    (
        {"spacetime": []},
        expect_value_error_with_message(
            "Input data must contain ['spacetime'] as dimensions."
        ),
    ),
    (
        {"time": [0, 100], "space": ["x", "y"]},
        expect_value_error_with_message(
            "Input data must contain [100] in the 'time' coordinates."
        ),
    ),
    (
        {"space": ["x", "y", "z"]},
        expect_value_error_with_message(
            "Input data must contain ['z'] in the 'space' coordinates."
        ),
    ),
]  # Invalid cases (raise ValueError)


@pytest.mark.parametrize(
    "required_dims_coords, expected_exception",
    valid_cases + invalid_cases,
)
def test_validate_dims_coords(
    valid_poses_dataset_uniform_linear_motion,  # fixture from conftest.py
    required_dims_coords,
    expected_exception,
):
    """Test validate_dims_coords for both valid and invalid inputs."""
    position_array = valid_poses_dataset_uniform_linear_motion["position"]
    with expected_exception:
        validate_dims_coords(position_array, required_dims_coords)
