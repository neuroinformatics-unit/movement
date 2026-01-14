import json
import re
from typing import Any

import cv2
import numpy as np
import pytest
import xarray as xr

from movement.transforms import compute_homography_transform, scale

SPATIAL_COORDS_2D = {"space": ["x", "y"]}
SPATIAL_COORDS_3D = {"space": ["x", "y", "z"]}


def nparray_0_to_23() -> np.ndarray:
    """Create a 2D nparray from 0 to 23."""
    return np.arange(0, 24).reshape(12, 2)


def data_array_with_dims_and_coords(
    data: np.ndarray,
    dims: list | tuple = ("time", "space"),
    coords: dict[str, list[str]] = SPATIAL_COORDS_2D,
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


def drop_attrs_log(attrs: dict) -> dict:
    """Drop the log string from attrs to faclitate testing.
    The log string will never exactly match, because datetimes differ.
    """
    attrs_copy = attrs.copy()
    if "log" in attrs:
        attrs_copy.pop("log", None)
    return attrs_copy


@pytest.fixture
def sample_data_2d() -> xr.DataArray:
    """Turn the nparray_0_to_23 into a DataArray."""
    return data_array_with_dims_and_coords(nparray_0_to_23())


@pytest.fixture
def sample_data_3d() -> xr.DataArray:
    """Turn the nparray_0_to_23 into a DataArray with 3D space."""
    return data_array_with_dims_and_coords(
        nparray_0_to_23().reshape(8, 3),
        coords=SPATIAL_COORDS_3D,
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
    sample_data_2d: xr.DataArray,
    optional_arguments: dict[str, Any],
    expected_output: xr.DataArray,
):
    """Test scaling with different factors and space_units."""
    scaled_data = scale(sample_data_2d, **optional_arguments)
    xr.testing.assert_equal(scaled_data, expected_output)
    assert drop_attrs_log(scaled_data.attrs) == expected_output.attrs


@pytest.mark.parametrize(
    "dims, data_shape",
    [
        (["time", "space"], (3, 2)),
        (["space", "time"], (2, 3)),
        (["time", "individual", "keypoint", "space"], (3, 6, 4, 2)),
        (["time", "individual", "keypoint", "space"], (2, 2, 2, 2)),
    ],
    ids=[
        "time-space",
        "space-time",
        "time-individual-keypoint-space",
        "2x2x2x2",
    ],
)
def test_scale_space_dimension(dims: list[str], data_shape):
    """Test scaling with transposed data along the correct dimension.

    The scaling factor should be broadcasted along the space axis irrespective
    of the order of the dimensions in the input data.
    """
    factor = [0.5, 2]
    numerical_data = np.arange(np.prod(data_shape)).reshape(data_shape)
    data = xr.DataArray(numerical_data, dims=dims, coords=SPATIAL_COORDS_2D)
    scaled_data = scale(data, factor=factor)
    broadcast_list = [1 if dim != "space" else len(factor) for dim in dims]
    expected_output_data = data * np.array(factor).reshape(broadcast_list)

    assert scaled_data.shape == data.shape
    xr.testing.assert_equal(scaled_data, expected_output_data)


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
    sample_data_2d: xr.DataArray,
    optional_arguments_1: dict[str, Any],
    optional_arguments_2: dict[str, Any],
    expected_output: xr.DataArray,
):
    """Test scaling when applied twice.
    The second scaling operation should update the space_unit attribute if
    provided, or remove it if None is passed explicitly or by default.
    """
    output_data_array = scale(
        scale(sample_data_2d, **optional_arguments_1),
        **optional_arguments_2,
    )
    xr.testing.assert_equal(output_data_array, expected_output)
    assert drop_attrs_log(output_data_array.attrs) == expected_output.attrs


@pytest.mark.parametrize(
    "invalid_factor, expected_error_message",
    [
        pytest.param(
            np.zeros((3, 3, 4)),
            "Factor must be an object that can be converted to a 1D numpy"
            " array, got 3D",
            id="3D factor",
        ),
        pytest.param(
            np.zeros(3),
            "Factor shape (3,) does not match the shape "
            "of the space dimension (2,)",
            id="space dimension mismatch",
        ),
    ],
)
def test_scale_value_error(
    sample_data_2d: xr.DataArray,
    invalid_factor: np.ndarray,
    expected_error_message: str,
):
    """Test invalid factors raise correct error type and message."""
    with pytest.raises(ValueError) as error:
        scale(sample_data_2d, factor=invalid_factor)
    assert str(error.value) == expected_error_message


@pytest.mark.parametrize(
    "factor", [2, [1, 2, 0.5]], ids=["uniform scaling", "multi-axis scaling"]
)
def test_scale_3d_space(factor, sample_data_3d: xr.DataArray):
    """Test scaling a DataArray with 3D space."""
    scaled_data = scale(sample_data_3d, factor=factor)
    expected_output = data_array_with_dims_and_coords(
        nparray_0_to_23().reshape(8, 3) * np.array(factor).reshape(1, -1),
        coords=SPATIAL_COORDS_3D,
    )
    xr.testing.assert_equal(scaled_data, expected_output)


@pytest.mark.parametrize(
    "factor",
    [2, [1, 2, 0.5]],
    ids=["uniform scaling", "multi-axis scaling"],
)
def test_scale_invalid_3d_space(factor):
    """Test scaling data with invalid 3D space coordinates."""
    invalid_coords = {"space": ["x", "flubble", "y"]}  # "z" is missing
    invalid_sample_data_3d = data_array_with_dims_and_coords(
        nparray_0_to_23().reshape(8, 3),
        coords=invalid_coords,
    )
    with pytest.raises(ValueError) as error:
        scale(invalid_sample_data_3d, factor=factor)
    assert str(error.value) == (
        "Input data must contain ['z'] in the 'space' coordinates.\n"
    )


def test_scale_log(sample_data_2d: xr.DataArray):
    """Test that the log attribute is correctly populated
    in the scaled data array.
    """

    def verify_log_entry(entry, expected_factor, expected_space_unit):
        """Verify each scale log entry."""
        assert entry["factor"] == expected_factor
        assert entry["space_unit"] == expected_space_unit
        assert entry["operation"] == "scale"
        assert "datetime" in entry

    # scale data twice
    scaled_data = scale(
        scale(sample_data_2d, factor=2, space_unit="elephants"),
        factor=[1, 2],
        space_unit="crabs",
    )

    # verify the log attribute
    assert "log" in scaled_data.attrs
    log_entries = json.loads(scaled_data.attrs["log"])
    assert len(log_entries) == 2
    verify_log_entry(log_entries[0], "2", "'elephants'")
    verify_log_entry(log_entries[1], "[1, 2]", "'crabs'")


@pytest.mark.parametrize(
    ["src_points", "dst_points"],
    [
        pytest.param(
            np.array([[1, 1], [5, 1], [5, 3], [1, 3]], dtype=np.float32),
            np.array([[1, 1], [5, 1], [5, 3], [1, 3]], dtype=np.float32),
            id="Identical rectangles",
        ),
        pytest.param(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            np.array(
                [
                    [3, -1],
                    [4.73205081, 0],
                    [3.98205081, 1.29903811],
                    [2.25000000, 0.299038106],
                ],
                dtype=np.float32,
            ),
            id="Rotated and scaled square",
        ),
        pytest.param(
            np.array(
                [[0, 0], [0.5, 0], [1, 0], [1.0, 0.5], [1, 1], [0, 1]],
                dtype=np.float32,
            ),
            np.array(
                [
                    [3, -1],
                    [3.8660254, -0.5],
                    [4.73205081, 0],
                    [4.35705081, 0.649519053],
                    [3.98205081, 1.29903811],
                    [2.25000000, 0.299038106],
                ],
                dtype=np.float32,
            ),
            id="Rotated and scaled square with collinear points",
        ),
        pytest.param(
            np.array(
                [
                    [0, 0],
                    [5.0e-01, 1.0e-05],
                    [0.5, 0],
                    [1, 0],
                    [1.0, 0.5],
                    [1, 1],
                    [0, 1],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [3, -1],
                    [3.86601790, -0.499987010],
                    [3.8660254, -0.5],
                    [4.73205081, 0],
                    [4.35705081, 0.649519053],
                    [3.98205081, 1.29903811],
                    [2.25000000, 0.299038106],
                ],
                dtype=np.float32,
            ),
            id="Rotated and scaled square with \
                  collinear and degenerate points",
        ),
    ],
)
def test_compute_homography_transform_happy_path(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> None:
    expected_transform, _ = cv2.findHomography(
        src_points, dst_points, method=cv2.RANSAC
    )
    computed_transform = compute_homography_transform(src_points, dst_points)
    assert np.allclose(computed_transform, expected_transform, atol=1e-6)


@pytest.mark.parametrize(
    ["src_points", "dst_points", "error"],
    [
        pytest.param(
            np.array([[1, 1, 3], [5, 1, 4]], dtype=np.float32),
            np.array([[1, 1, 4], [5, 1, 5]], dtype=np.float32),
            ValueError("Points must be 2-dimensional."),
            id="Using 3-D points",
        ),
        pytest.param(
            np.array([[[1, 1], [4, 5]], [[5, 1], [4, 5]]], dtype=np.float32),
            np.array([[1, 1], [5, 1]], dtype=np.float32),
            ValueError("Points must be 2-dimensional arrays."),
            id="Using 3-D source array",
        ),
        pytest.param(
            np.array([[1, 1], [5, 1]], dtype=np.float32),
            np.array([[[1, 1], [4, 5]], [[5, 1], [4, 5]]], dtype=np.float32),
            ValueError("Points must be 2-dimensional arrays."),
            id="Using 3-D destination array",
        ),
        pytest.param(
            np.array([[1, 1], [5, 1], [5, 3]], dtype=np.float32),
            np.array([[1, 1], [5, 1]], dtype=np.float32),
            ValueError(
                "Source and destination points must have the same shape."
            ),
            id="Using more source points than destination points",
        ),
        pytest.param(
            np.array([[1, 1], [5, 1]], dtype=np.float32),
            np.array([[1, 1], [5, 1], [5, 3]], dtype=np.float32),
            ValueError(
                "Source and destination points must have the same shape."
            ),
            id="Using more destination points than source points",
        ),
        pytest.param(
            np.array([[1, 1, 3], [5, 1, 5]], dtype=np.float32),
            np.array([[1, 1], [5, 1]], dtype=np.float32),
            ValueError(
                "Source and destination points must have the same shape."
            ),
            id="Using different source points than destination points",
        ),
        pytest.param(
            np.array([[1, 1], [5, 1], [5, 3]], dtype=np.float32),
            np.array([[1, 1], [5, 1], [5, 3]], dtype=np.float32),
            ValueError(
                "Insufficient points to compute the homography transformation."
            ),
            id="Insufficient points",
        ),
        pytest.param(
            np.array(
                [[1, 1], [5, 1], [5, 3], [4.999999, 3]], dtype=np.float32
            ),
            np.array(
                [[1, 1], [5, 1], [5, 3], [4.999999, 3]], dtype=np.float32
            ),
            ValueError(
                "Insufficient points to compute the homography transformation."
            ),
            id="Insufficient points due to degeneracy",
        ),
        pytest.param(
            np.array([[1, 1], [5, 1], [5, 3], [5, 2]], dtype=np.float32),
            np.array([[1, 1], [5, 1], [5, 3], [5, 2]], dtype=np.float32),
            ValueError(
                "Insufficient points to compute the homography transformation."
            ),
            id="Insufficient points due to collinearity",
        ),
    ],
)
def test_compute_homography_transform_invalid_input(
    src_points: np.ndarray, dst_points: np.ndarray, error: ValueError
) -> None:
    with pytest.raises(type(error), match=re.escape(str(error))):
        compute_homography_transform(src_points, dst_points)
