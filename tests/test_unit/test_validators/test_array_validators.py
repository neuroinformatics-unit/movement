import re
from contextlib import nullcontext as does_not_raise

import pytest

from movement.validators.arrays import validate_dims_coords


def expect_value_error_with_message(error_msg):
    """Expect a ValueError with the specified error message."""
    return pytest.raises(ValueError, match=re.escape(error_msg))


# dims_coords: dict, exact_coords: bool, expected_exception
valid_cases = [
    ({"time": []}, False, does_not_raise()),
    ({"time": []}, True, does_not_raise()),
    ({"time": [0, 1]}, False, does_not_raise()),
    ({"space": ["x", "y"]}, False, does_not_raise()),
    ({"space": ["x", "y"]}, True, does_not_raise()),
    ({"time": [], "space": []}, False, does_not_raise()),
    ({"time": [], "space": ["x", "y"]}, False, does_not_raise()),
]  # Valid cases (no error)

invalid_cases = [
    (
        {"spacetime": []},
        False,
        expect_value_error_with_message(
            "Input data must contain ['spacetime'] as dimensions."
        ),
    ),
    (
        {"time": [0, 100], "space": ["x", "y"]},
        False,
        expect_value_error_with_message(
            "Input data must contain [100] in the 'time' coordinates."
        ),
    ),
    (
        {"space": ["x", "y", "z"]},
        False,
        expect_value_error_with_message(
            "Input data must contain ['z'] in the 'space' coordinates."
        ),
    ),
    (
        {"space": ["x"]},
        True,
        expect_value_error_with_message(
            "Dimension 'space' must only contain ['x'] as coordinates, "
        ),
    ),
]  # Invalid cases (raise ValueError)


@pytest.mark.parametrize(
    "required_dims_coords, exact_coords, expected_exception",
    valid_cases + invalid_cases,
)
def test_validate_dims_coords(
    valid_poses_dataset,  # fixture from conftest.py
    required_dims_coords,
    exact_coords,
    expected_exception,
):
    """Test validate_dims_coords for both valid and invalid inputs."""
    position_array = valid_poses_dataset["position"]
    with expected_exception:
        validate_dims_coords(
            position_array, required_dims_coords, exact_coords=exact_coords
        )
