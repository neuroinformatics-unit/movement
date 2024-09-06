import re

import pytest
import xarray as xr

from movement.validators.arrays import validate_dims_coords


@pytest.fixture
def valid_data_array():
    return xr.DataArray(
        data=[[1, 2], [3, 4]],
        dims=["time", "space"],
        coords={"time": [0, 1], "space": ["x", "y"]},
    )


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
    valid_data_array,
    required_dims_coords,
):
    """Test that valid inputs do not raise an error."""
    validate_dims_coords(valid_data_array, required_dims_coords)


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
    valid_data_array,
    required_dims_coords,
    expected_error_message,
):
    """Test that invalid inputs raise a ValueError with expected message."""
    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        validate_dims_coords(valid_data_array, required_dims_coords)
