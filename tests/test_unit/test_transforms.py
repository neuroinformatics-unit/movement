from typing import Any

import numpy as np
import pytest
import xarray as xr

from movement.transforms import scale


def nparray_0_to_23() -> np.ndarray:
    return np.arange(0, 24).reshape(12, 2)


@pytest.fixture
def sample_data_array() -> xr.DataArray:
    """Turn the nparray_0_to_23 into a DataArray."""
    return xr.DataArray(
        nparray_0_to_23(),
        dims=["time", "space"],
        coords={"space": ["x", "y"]},
    )


@pytest.mark.parametrize(
    ["optional_arguments", "expected_output"],
    [
        pytest.param(
            {},
            nparray_0_to_23(),
            id="Do nothing",
        ),
        pytest.param(
            {"unit": "elephants"},
            xr.DataArray(
                nparray_0_to_23(),
                dims=["time", "space"],
                coords={"space": ["x", "y"]},
                attrs={"unit": "elephants"},
            ),
            id="Add example unit",
        ),
        pytest.param(
            {"factor": 2},
            nparray_0_to_23() * 2,
            id="Double",
        ),
        pytest.param(
            {"factor": 0.5},
            nparray_0_to_23() * 0.5,
            id="Halve",
        ),
        pytest.param(
            {"factor": 0.5, "unit": "elephants"},
            xr.DataArray(
                nparray_0_to_23() * 0.5,
                dims=["time", "space"],
                coords={"space": ["x", "y"]},
                attrs={"unit": "elephants"},
            ),
            id="Halve and add example unit",
        ),
    ],
)
def test_scale(
    sample_data_array: xr.DataArray,
    optional_arguments: dict[str, Any],
    expected_output: xr.DataArray,
):
    if isinstance(expected_output, np.ndarray):
        expected_output = xr.DataArray(
            expected_output,
            dims=["time", "space"],
            coords={"space": ["x", "y"]},
        )

    output_data_array = scale(sample_data_array, **optional_arguments)
    xr.testing.assert_equal(output_data_array, expected_output)
