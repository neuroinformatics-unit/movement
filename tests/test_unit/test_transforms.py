import json
import re
from typing import Any

import cv2
import numpy as np
import pytest
import xarray as xr

from movement.transforms import (
    compute_homography_transform,
    poses_to_bboxes,
    scale,
)

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
        (["time", "individuals", "keypoints", "space"], (3, 6, 4, 2)),
        (["time", "individuals", "keypoints", "space"], (2, 2, 2, 2)),
    ],
    ids=[
        "time-space",
        "space-time",
        "time-individuals-keypoints-space",
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


# ============= Tests for poses_to_bboxes =============


def create_poses_dataset(
    n_frames=5,
    n_keypoints=3,
    n_individuals=2,
    keypoint_positions=None,
    confidence_values=None,
    ds_attrs=None,
):
    """Create poses datasets with configurable parameters.

    Args:
        n_frames: Number of time frames
        n_keypoints: Number of keypoints per individual
        n_individuals: Number of individuals
        keypoint_positions: Dict mapping (individual_id, keypoint_id) to
            (x, y) tuples
        confidence_values: Array of confidence values or None
        ds_attrs: Dictionary of dataset attributes

    Returns:
        xarray.Dataset with pose data

    """
    n_space = 2
    position = np.zeros((n_frames, n_space, n_keypoints, n_individuals))

    # Set default positions if not provided
    if keypoint_positions is None:
        # Individual 0: keypoints at (0,0), (10,0), (10,10)
        # Individual 1: keypoints at (20,20), (30,20), (30,30)
        if n_individuals > 0 and n_keypoints >= 3:
            position[:, 0, 0, 0] = 0.0
            position[:, 1, 0, 0] = 0.0
            position[:, 0, 1, 0] = 10.0
            position[:, 1, 1, 0] = 0.0
            position[:, 0, 2, 0] = 10.0
            position[:, 1, 2, 0] = 10.0
        if n_individuals > 1 and n_keypoints >= 3:
            position[:, 0, 0, 1] = 20.0
            position[:, 1, 0, 1] = 20.0
            position[:, 0, 1, 1] = 30.0
            position[:, 1, 1, 1] = 20.0
            position[:, 0, 2, 1] = 30.0
            position[:, 1, 2, 1] = 30.0
    else:
        # Apply custom positions
        for (ind_id, kpt_id), (x, y) in keypoint_positions.items():
            position[:, 0, kpt_id, ind_id] = x
            position[:, 1, kpt_id, ind_id] = y

    data_vars = {
        "position": xr.DataArray(
            position,
            dims=("time", "space", "keypoints", "individuals"),
        ),
    }

    # Add confidence if provided
    if confidence_values is not None:
        data_vars["confidence"] = xr.DataArray(
            confidence_values,
            dims=("time", "keypoints", "individuals"),
        )

    # Set default attributes
    if ds_attrs is None:
        ds_attrs = {"fps": 30, "time_unit": "frames", "ds_type": "poses"}

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.arange(n_frames),
            "space": ["x", "y"],
            "keypoints": [f"kpt_{i}" for i in range(n_keypoints)],
            "individuals": [f"id_{i}" for i in range(n_individuals)],
        },
        attrs=ds_attrs,
    )

    return ds


def verify_bbox_result(
    result, expected_position, expected_shape, ind_id=0, frame_id=0
):
    """Verify bbox position and shape values.

    Args:
        result: The result dataset from poses_to_bboxes
        expected_position: Tuple of (x_centroid, y_centroid)
        expected_shape: Tuple of (width, height)
        ind_id: Individual index to check
        frame_id: Frame index to check

    """
    assert np.allclose(
        result.position.values[frame_id, 0, ind_id], expected_position[0]
    )
    assert np.allclose(
        result.position.values[frame_id, 1, ind_id], expected_position[1]
    )
    assert np.allclose(
        result.shape.values[frame_id, 0, ind_id], expected_shape[0]
    )
    assert np.allclose(
        result.shape.values[frame_id, 1, ind_id], expected_shape[1]
    )


@pytest.fixture
def simple_poses_dataset():
    """Create a simple poses dataset.

    Creates a dataset with 2 individuals, 3 keypoints, 5 frames.
    """
    confidence = np.full((5, 3, 2), 0.9)
    return create_poses_dataset(
        n_frames=5,
        n_keypoints=3,
        n_individuals=2,
        confidence_values=confidence,
    )


def test_poses_to_bboxes_basic(simple_poses_dataset):
    """Test basic conversion from poses to bboxes."""
    result = poses_to_bboxes(simple_poses_dataset)

    # Check dimensions
    assert result.dims == {"time": 5, "space": 2, "individuals": 2}

    # Check data variables exist
    assert "position" in result.data_vars
    assert "shape" in result.data_vars
    assert "confidence" in result.data_vars

    # Check coordinates
    assert list(result.coords["space"].values) == ["x", "y"]
    assert list(result.coords["individuals"].values) == ["id_0", "id_1"]

    # Check metadata
    assert result.attrs["ds_type"] == "bboxes"
    assert "log" in result.attrs
    assert result.attrs["fps"] == 30

    # Verify bbox calculations for individual 0
    # Keypoints: (0,0), (10,0), (10,10)
    # -> bbox centroid (5, 5), shape (10, 10)
    verify_bbox_result(result, (5.0, 5.0), (10.0, 10.0), ind_id=0)

    # Verify bbox calculations for individual 1
    # Keypoints: (20,20), (30,20), (30,30)
    # -> bbox centroid (25, 25), shape (10, 10)
    verify_bbox_result(result, (25.0, 25.0), (10.0, 10.0), ind_id=1)

    # Verify confidence (should be mean of keypoint confidences = 0.9)
    assert np.allclose(result.confidence.values[0, 0], 0.9)
    assert np.allclose(result.confidence.values[0, 1], 0.9)


@pytest.mark.parametrize("padding_px", [0, 5, 10.5, 100])
def test_poses_to_bboxes_with_padding(simple_poses_dataset, padding_px):
    """Test bounding box calculation with various padding values."""
    result = poses_to_bboxes(simple_poses_dataset, padding_px=padding_px)

    # Individual 0 has width=10, height=10 before padding
    expected_width = 10.0 + 2 * padding_px
    expected_height = 10.0 + 2 * padding_px

    assert np.allclose(result.shape.values[0, 0, 0], expected_width)
    assert np.allclose(result.shape.values[0, 1, 0], expected_height)


def test_poses_to_bboxes_single_keypoint():
    """Test conversion with single keypoint per individual.

    With a single keypoint, the bbox should have zero width/height.
    """
    # Single keypoint at (5, 10)
    ds = create_poses_dataset(
        n_frames=3,
        n_keypoints=1,
        n_individuals=1,
        keypoint_positions={(0, 0): (5.0, 10.0)},
    )

    result = poses_to_bboxes(ds)

    # Centroid should be at the single keypoint
    assert np.allclose(result.position.values[:, 0, 0], 5.0)
    assert np.allclose(result.position.values[:, 1, 0], 10.0)

    # Shape should be zero (no span)
    assert np.allclose(result.shape.values[:, 0, 0], 0.0)  # width
    assert np.allclose(result.shape.values[:, 1, 0], 0.0)  # height


def test_poses_to_bboxes_with_nan(simple_poses_dataset):
    """Test conversion with NaN values in positions."""
    ds = simple_poses_dataset.copy(deep=True)

    # Set some keypoints to NaN
    ds.position.values[0, :, 0, 0] = np.nan  # First keypoint of ind 0, frame 0
    ds.position.values[1, :, 1, 1] = (
        np.nan
    )  # Second keypoint of ind 1, frame 1

    result = poses_to_bboxes(ds)

    # Frame 0, individual 0: only 2 keypoints valid
    # -> bbox should still be computed
    # Remaining keypoints: (10,0), (10,10)
    # -> centroid (10, 5), shape (0, 10)
    verify_bbox_result(result, (10.0, 5.0), (0.0, 10.0), ind_id=0, frame_id=0)

    # Frame 1, individual 1: only 2 keypoints valid
    # Remaining keypoints: (20,20), (30,30)
    # -> centroid (25, 25), shape (10, 10)
    assert np.allclose(result.position.values[1, 0, 1], 25.0)
    assert np.allclose(result.position.values[1, 1, 1], 25.0)


def test_poses_to_bboxes_all_nan_frame():
    """Test frame where all keypoints are NaN for an individual."""
    ds = create_poses_dataset(
        n_frames=3,
        n_keypoints=3,
        n_individuals=1,
        keypoint_positions={(0, i): (5.0, 5.0) for i in range(3)},
    )
    # Set all keypoints to NaN in frame 1
    ds.position.values[1, :, :, 0] = np.nan

    result = poses_to_bboxes(ds)

    # Frame 1 should have all NaN
    assert np.isnan(result.position.values[1, 0, 0])
    assert np.isnan(result.position.values[1, 1, 0])
    assert np.isnan(result.shape.values[1, 0, 0])
    assert np.isnan(result.shape.values[1, 1, 0])
    assert np.isnan(result.confidence.values[1, 0])

    # Other frames should be valid
    # (all keypoints at 5,5 -> bbox at 5,5 with 0 size)
    assert np.allclose(result.position.values[0, 0, 0], 5.0)
    assert np.allclose(result.position.values[2, 0, 0], 5.0)


def test_poses_to_bboxes_partial_nan_keypoints():
    """Test keypoints with some coordinates NaN (x valid, y NaN)."""
    position = np.array(
        [
            [
                [[1.0], [2.0], [3.0]],  # x coords frame 0
                [
                    [1.0],
                    [np.nan],
                    [3.0],
                ],  # y coords frame 0 - middle keypoint has NaN y
            ],
            [
                [[1.0], [2.0], [3.0]],  # x coords frame 1
                [[1.0], [2.0], [3.0]],  # y coords frame 1 - all valid
            ],
        ]
    )

    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                position,
                dims=("time", "space", "keypoints", "individuals"),
            ),
        },
        coords={
            "time": np.arange(2),
            "space": ["x", "y"],
            "keypoints": ["kpt_0", "kpt_1", "kpt_2"],
            "individuals": ["id_0"],
        },
        attrs={"ds_type": "poses"},
    )

    result = poses_to_bboxes(ds)

    # Frame 0: middle keypoint should be ignored (has NaN in y)
    # Valid keypoints: (1,1), (3,3) -> centroid (2, 2), shape (2, 2)
    assert np.allclose(result.position.values[0, 0, 0], 2.0)
    assert np.allclose(result.position.values[0, 1, 0], 2.0)
    assert np.allclose(result.shape.values[0, 0, 0], 2.0)
    assert np.allclose(result.shape.values[0, 1, 0], 2.0)

    # Frame 1: all keypoints valid
    # Keypoints: (1,1), (2,2), (3,3) -> centroid (2, 2), shape (2, 2)
    assert np.allclose(result.position.values[1, 0, 0], 2.0)
    assert np.allclose(result.position.values[1, 1, 0], 2.0)


def test_poses_to_bboxes_confidence_aggregation(simple_poses_dataset):
    """Test that confidence is correctly aggregated (mean)."""
    # Modify confidence values
    ds = simple_poses_dataset.copy(deep=True)
    ds.confidence.values[0, 0, 0] = 0.6
    ds.confidence.values[0, 1, 0] = 0.8
    ds.confidence.values[0, 2, 0] = 1.0
    # Mean should be (0.6 + 0.8 + 1.0) / 3 = 0.8

    result = poses_to_bboxes(ds)

    assert np.allclose(result.confidence.values[0, 0], 0.8)


def test_poses_to_bboxes_no_confidence():
    """Test conversion when confidence is not in dataset."""
    ds = create_poses_dataset(
        n_frames=2,
        n_keypoints=2,
        n_individuals=1,
        keypoint_positions={(0, i): (5.0, 5.0) for i in range(2)},
        confidence_values=None,
    )

    result = poses_to_bboxes(ds)

    # Confidence should be all NaN
    assert np.all(np.isnan(result.confidence.values))


def test_poses_to_bboxes_nan_confidence_values():
    """Test when some confidence values are NaN."""
    confidence = np.tile(np.array([[[0.8], [np.nan], [0.6]]]), (2, 1, 1))
    ds = create_poses_dataset(
        n_frames=2,
        n_keypoints=3,
        n_individuals=1,
        keypoint_positions={(0, i): (5.0, 5.0) for i in range(3)},
        confidence_values=confidence,
    )

    result = poses_to_bboxes(ds)

    # Mean should ignore NaN: (0.8 + 0.6) / 2 = 0.7
    assert np.allclose(result.confidence.values[0, 0], 0.7)


def test_poses_to_bboxes_preserves_metadata(simple_poses_dataset):
    """Test that metadata is preserved in conversion."""
    result = poses_to_bboxes(simple_poses_dataset)

    # Check original attributes are preserved
    assert result.attrs["fps"] == 30
    assert result.attrs["time_unit"] == "frames"

    # Check ds_type is updated
    assert result.attrs["ds_type"] == "bboxes"

    # Check log attribute exists
    assert "log" in result.attrs


def test_poses_to_bboxes_preserves_coords(simple_poses_dataset):
    """Test that time and individual coordinates are preserved."""
    result = poses_to_bboxes(simple_poses_dataset)

    # Check time coordinates
    np.testing.assert_array_equal(
        result.coords["time"].values,
        simple_poses_dataset.coords["time"].values,
    )

    # Check individual names
    assert list(result.coords["individuals"].values) == ["id_0", "id_1"]


# ============= Error Handling Tests =============


def test_poses_to_bboxes_3d_poses():
    """Test that 3D poses raise appropriate error."""
    position = np.full((2, 3, 2, 1), 5.0)
    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                position,
                dims=("time", "space", "keypoints", "individuals"),
            ),
        },
        coords={
            "time": np.arange(2),
            "space": ["x", "y", "z"],
            "keypoints": ["kpt_0", "kpt_1"],
            "individuals": ["id_0"],
        },
        attrs={"ds_type": "poses"},
    )

    with pytest.raises(
        ValueError,
        match="Input dataset must contain 2D poses only",
    ):
        poses_to_bboxes(ds)


def test_poses_to_bboxes_missing_position():
    """Test error when position data variable is missing."""
    ds = xr.Dataset(
        data_vars={
            "confidence": xr.DataArray(
                np.ones((2, 2, 1)),
                dims=("time", "keypoints", "individuals"),
            ),
        },
        coords={
            "time": [0, 1],
            "keypoints": ["kpt_0", "kpt_1"],
            "individuals": ["id_0"],
        },
    )

    with pytest.raises(
        ValueError,
        match="Input dataset must contain 'position' data variable",
    ):
        poses_to_bboxes(ds)


def test_poses_to_bboxes_invalid_input_type():
    """Test error with non-Dataset input."""
    data_array = xr.DataArray(np.ones((2, 2)), dims=("time", "space"))

    with pytest.raises(TypeError, match="Input must be an xarray.Dataset"):
        poses_to_bboxes(data_array)


def test_poses_to_bboxes_negative_padding():
    """Test that negative padding raises error."""
    ds = create_poses_dataset(
        n_frames=2,
        n_keypoints=2,
        n_individuals=1,
        keypoint_positions={(0, i): (5.0, 5.0) for i in range(2)},
    )

    with pytest.raises(ValueError, match="padding_px must be non-negative"):
        poses_to_bboxes(ds, padding_px=-5)


@pytest.mark.parametrize(
    "invalid_padding",
    ["10", [10], None],
    ids=["string", "list", "None"],
)
def test_poses_to_bboxes_invalid_padding_type(
    simple_poses_dataset, invalid_padding
):
    """Test that invalid padding types raise error."""
    with pytest.raises(TypeError, match="padding_px must be a number"):
        poses_to_bboxes(simple_poses_dataset, padding_px=invalid_padding)


def test_poses_to_bboxes_missing_dimensions():
    """Test error when position is missing required dimensions."""
    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                np.zeros((2,)),
                dims=("space",),
                coords={"space": ["x", "y"]},
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="Missing:",
    ):
        poses_to_bboxes(ds)


def test_poses_to_bboxes_degenerate_bbox():
    """Test when all keypoints are at same position (zero size bbox)."""
    # All keypoints at (5, 5)
    ds = create_poses_dataset(
        n_frames=2,
        n_keypoints=3,
        n_individuals=1,
        keypoint_positions={(0, i): (5.0, 5.0) for i in range(3)},
    )

    result = poses_to_bboxes(ds)

    # Centroid should be at (5, 5), shape should be zero
    for frame_id in range(2):
        verify_bbox_result(
            result, (5.0, 5.0), (0.0, 0.0), ind_id=0, frame_id=frame_id
        )


def test_poses_to_bboxes_log_attribute(simple_poses_dataset):
    """Test that log attribute is correctly updated."""
    import json

    result = poses_to_bboxes(simple_poses_dataset, padding_px=5)

    # Check log exists and contains the operation
    assert "log" in result.attrs
    log_str = result.attrs["log"]

    # Parse the JSON log string
    log_entries = json.loads(log_str)

    # Should have at least one entry
    assert len(log_entries) > 0

    # Parse the log entry (it's a list of dicts)
    last_entry = log_entries[-1]
    assert last_entry["operation"] == "poses_to_bboxes"
    assert "5" in str(last_entry["padding_px"]) or "5.0" in str(
        last_entry["padding_px"]
    )
