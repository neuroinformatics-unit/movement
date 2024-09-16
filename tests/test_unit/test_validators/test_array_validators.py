import re
from contextlib import nullcontext as does_not_raise

import pytest

from movement.validators.arrays import validate_dims_coords


@pytest.mark.parametrize(
    "required_dims_coords, expected_exception, expected_error_message",
    [
        # Valid cases (no error)
        ({"time": []}, does_not_raise(), None),
        ({"time": [0, 1]}, does_not_raise(), None),
        ({"space": ["x", "y"]}, does_not_raise(), None),
        ({"time": [], "space": []}, does_not_raise(), None),
        ({"time": [], "space": ["x", "y"]}, does_not_raise(), None),
        # Invalid cases (raise ValueError)
        (
            {"spacetime": []},
            pytest.raises(ValueError),
            "Input data must contain ['spacetime'] as dimensions.",
        ),
        (
            {"time": [0, 100], "space": ["x", "y"]},
            pytest.raises(ValueError),
            "Input data must contain [100] in the 'time' coordinates.",
        ),
        (
            {"space": ["x", "y", "z"]},
            pytest.raises(ValueError),
            "Input data must contain ['z'] in the 'space' coordinates.",
        ),
    ],
)
def test_validate_dims_coords(
    valid_poses_dataset_uniform_linear_motion,  # fixture from conftest.py
    required_dims_coords,
    expected_exception,
    expected_error_message,
):
    """Test validate_dims_coords for both valid and invalid inputs."""
    position_array = valid_poses_dataset_uniform_linear_motion["position"]
    with expected_exception as exc_info:
        validate_dims_coords(position_array, required_dims_coords)
        if expected_error_message:
            assert re.search(
                re.escape(expected_error_message), str(exc_info.value)
            )
