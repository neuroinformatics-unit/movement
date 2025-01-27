from typing import Any

import numpy as np
import pytest
import xarray as xr

from movement.transforms import scale

DEFAULT_SPATIAL_COORDS = {"space": ["x", "y"]}


def nparray_0_to_23() -> np.ndarray:
    """Create a 2D nparray from 0 to 23."""
    return np.arange(0, 24).reshape(12, 2)


@pytest.fixture
def sample_data() -> xr.DataArray:
    """Turn the nparray_0_to_23 into a DataArray."""
    return data_array_with_dims_and_coords(nparray_0_to_23())


def data_array_with_dims_and_coords(
    data: np.ndarray,
    dims: list | tuple = ("time", "space"),
    coords: dict[str, list[str]] = DEFAULT_SPATIAL_COORDS,
    **attributes: Any,
) -> xr.DataArray:
    """Create a DataArray with given data, dimensions, coordinates, and
    attributes (e.g. space_unit or factor).
    """
    return xr.DataArray(
        data,
        dims=dims,
        coords=coords,
        attrs=attributes,
    )


@pytest.mark.parametrize(
    ["optional_arguments", "expected_output"],
    [
        pytest.param(
            {},
            data_array_with_dims_and_coords(nparray_0_to_23()),
            id="Do nothing",
        ),
        pytest.param(
            {"space_unit": "elephants"},
            data_array_with_dims_and_coords(
                nparray_0_to_23(), space_unit="elephants"
            ),
            id="No scaling, add space_unit",
        ),
        pytest.param(
            {"factor": 2},
            data_array_with_dims_and_coords(nparray_0_to_23() * 2),
            id="Double, no space_unit",
        ),
        pytest.param(
            {"factor": 0.5},
            data_array_with_dims_and_coords(nparray_0_to_23() * 0.5),
            id="Halve, no space_unit",
        ),
        pytest.param(
            {"factor": 0.5, "space_unit": "elephants"},
            data_array_with_dims_and_coords(
                nparray_0_to_23() * 0.5, space_unit="elephants"
            ),
            id="Halve, add space_unit",
        ),
        pytest.param(
            {"factor": [0.5, 2]},
            data_array_with_dims_and_coords(
                nparray_0_to_23() * [0.5, 2],
            ),
            id="x / 2, y * 2",
        ),
        pytest.param(
            {"factor": np.array([0.5, 2]).reshape(1, 2)},
            data_array_with_dims_and_coords(
                nparray_0_to_23() * [0.5, 2],
            ),
            id="x / 2, y * 2, should squeeze to cast across space",
        ),
    ],
)
def test_scale(
    sample_data: xr.DataArray,
    optional_arguments: dict[str, Any],
    expected_output: xr.DataArray,
):
    """Test scaling with different factors and space_units."""
    scaled_data = scale(sample_data, **optional_arguments)
    xr.testing.assert_equal(scaled_data, expected_output)
    assert scaled_data.attrs == expected_output.attrs


def test_scale_inverted_data():
    """Test scaling with transposed data along the correct dimension.

    The factor is reshaped to (1, 1, 4, 1) so that it can be broadcasted along
    the third dimension ("y") which matches the length of the scaling factor.
    """
    factor = [0.5, 2]
    transposed_data = data_array_with_dims_and_coords(
        nparray_0_to_23().transpose(), dims=["space", "time"]
    )
    output_array = scale(transposed_data, factor=factor)
    expected_output = data_array_with_dims_and_coords(
        (nparray_0_to_23() * factor).transpose(), dims=["space", "time"]
    )
    xr.testing.assert_equal(output_array, expected_output)

    factor = [0.1, 0.2, 0.3, 0.4]
    data_shape = (3, 5, 4, 2)
    numerical_data = np.arange(np.prod(data_shape)).reshape(data_shape)
    input_data = xr.DataArray(numerical_data, dims=["w", "x", "y", "z"])
    output_array = scale(input_data, factor=factor)
    assert output_array.shape == input_data.shape
    xr.testing.assert_equal(
        output_array, input_data * np.array(factor).reshape(1, 1, 4, 1)
    )


def test_scale_first_matching_axis():
    """Test scaling when multiple axes match the scaling factor's length.
    The scaling factor should be broadcasted along the first matching axis.
    """
    factor = [0.5, 1]
    data_shape = (2, 2)
    numerical_data = np.arange(np.prod(data_shape)).reshape(data_shape)
    input_data = xr.DataArray(numerical_data, dims=["x", "y"])
    output_array = scale(input_data, factor=factor)
    assert output_array.shape == input_data.shape
    assert np.isclose(input_data.values[0] * 0.5, output_array.values[0]).all()
    assert np.isclose(input_data.values[1], output_array.values[1]).all()


@pytest.mark.parametrize(
    ["optional_arguments_1", "optional_arguments_2", "expected_output"],
    [
        pytest.param(
            {"factor": 2, "space_unit": "elephants"},
            {"factor": 0.5, "space_unit": "crabs"},
            data_array_with_dims_and_coords(
                nparray_0_to_23(), space_unit="crabs"
            ),
            id="No net scaling, final crabs space_unit",
        ),
        pytest.param(
            {"factor": 2, "space_unit": "elephants"},
            {"factor": 0.5, "space_unit": None},
            data_array_with_dims_and_coords(nparray_0_to_23()),
            id="No net scaling, no final space_unit",
        ),
        pytest.param(
            {"factor": 2, "space_unit": None},
            {"factor": 0.5, "space_unit": "elephants"},
            data_array_with_dims_and_coords(
                nparray_0_to_23(), space_unit="elephants"
            ),
            id="No net scaling, final elephant space_unit",
        ),
    ],
)
def test_scale_twice(
    sample_data: xr.DataArray,
    optional_arguments_1: dict[str, Any],
    optional_arguments_2: dict[str, Any],
    expected_output: xr.DataArray,
):
    """Test scaling when applied twice.
    The second scaling operation should update the space_unit attribute if
    provided, or remove it if None is passed explicitly or by default.
    """
    output_data_array = scale(
        scale(sample_data, **optional_arguments_1),
        **optional_arguments_2,
    )
    xr.testing.assert_equal(output_data_array, expected_output)
    assert output_data_array.attrs == expected_output.attrs


@pytest.mark.parametrize(
    "invalid_factor, expected_error_message",
    [
        (
            np.zeros((3, 3, 4)),
            "Factor must be an object that can be converted to a 1D numpy"
            " array, got 3D",
        ),
        (
            np.zeros(3),
            "Factor shape (3,) does not match the length of"
            " any data axes: (12, 2)",
        ),
    ],
)
def test_scale_value_error(
    sample_data: xr.DataArray,
    invalid_factor: np.ndarray,
    expected_error_message: str,
):
    """Test invalid factors raise correct error type and message."""
    with pytest.raises(ValueError) as error:
        scale(sample_data, factor=invalid_factor)
    assert str(error.value) == expected_error_message
