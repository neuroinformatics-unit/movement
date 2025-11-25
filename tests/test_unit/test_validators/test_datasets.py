import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from movement.validators.datasets import (
    ValidPosesDataset,
    _BaseValidDataset,
    _convert_fps_to_none_if_invalid,
    _convert_to_list_of_str,
)


@pytest.mark.parametrize(
    "input, expected_context, expected_output",
    [
        (
            "a",
            pytest.warns(
                UserWarning,
                match="Invalid value .* Expected a list of strings",
            ),
            ["a"],
        ),
        (("a", "b", "c"), does_not_raise(), ["a", "b", "c"]),
        ([1, 2, 3], does_not_raise(), ["1", "2", "3"]),
        (
            5,
            pytest.raises(ValueError, match="Expected a list of strings"),
            None,
        ),
    ],
)
def test_convert_to_list_of_str(input, expected_context, expected_output):
    """Test conversion of various inputs to list of strings."""
    with expected_context:
        out = _convert_to_list_of_str(input)
        assert out == expected_output


@pytest.mark.parametrize(
    "input, expected_context, expected_output",
    [
        (
            -1,
            pytest.warns(UserWarning, match="Invalid .* Setting fps to None"),
            None,
        ),
        (
            0,
            pytest.warns(UserWarning, match="Invalid .* Setting fps to None"),
            None,
        ),
        (10.0, does_not_raise(), 10.0),
        (10, does_not_raise(), 10.0),
        (None, does_not_raise(), None),
    ],
)
def test_convert_fps_to_none_if_invalid(
    input, expected_context, expected_output
):
    """Test handling of valid and invalid fps values."""
    with expected_context:
        out = _convert_fps_to_none_if_invalid(input)
    assert out == expected_output


def make_stub_dataset_class(
    dim_names=("time", "space"), var_names=("position",)
):
    """Create a minimal concrete subclass of _BaseValidDataset."""

    class StubDataset(_BaseValidDataset):
        """Minimal concrete subclass for testing _BaseValidDataset."""

        DIM_NAMES = dim_names
        VAR_NAMES = var_names
        _ALLOWED_SPACE_DIM_SIZE = 2  # 2D positions

        def _validate_individual_names_impl(self, attribute, value):
            pass

    return StubDataset


class StubAttr:
    """Stub for attrs.Attribute."""

    def __init__(self, name="stub_attribute"):
        """Initialise with a name."""
        self.name = name


def test_required_fields_missing():
    """Test that TypeError is raised when required fields are missing."""
    stub_dataset_class = make_stub_dataset_class()
    with pytest.raises(
        TypeError,
        match="missing 1 required keyword-only argument: 'position_array'",
    ):
        stub_dataset_class()


@pytest.mark.parametrize(
    "dim_names, position_array, expected_confidence, expected_ind_names",
    [
        (("time", "space"), np.zeros((5, 2)), np.full(5, np.nan), None),
        (
            ("time", "space", "individuals"),
            np.zeros((5, 2, 1)),
            np.full((5, 1), np.nan),
            ["id_0"],
        ),
        (
            ("time", "space", "individuals"),
            np.zeros((5, 2, 3)),
            np.full((5, 3), np.nan),
            ["id_0", "id_1", "id_2"],
        ),
    ],
)
def test_optional_fields_defaults(
    dim_names, position_array, expected_confidence, expected_ind_names
):
    """Test default values for optional fields."""
    stub_dataset_class = make_stub_dataset_class(dim_names=dim_names)
    ds = stub_dataset_class(position_array=position_array)
    np.testing.assert_allclose(
        ds.confidence_array, expected_confidence, equal_nan=True
    )
    assert ds.individual_names == expected_ind_names
    assert ds.fps is None
    assert ds.source_software is None


@pytest.mark.parametrize(
    "position_array, expected_error_message",
    [
        (np.zeros((5, 2, 3)), "have 2 dimensions, but got 3"),
        (np.zeros((5, 3)), "have 2 spatial dimensions, but got 3"),
    ],
    ids=[
        "Unexpected additional individuals dimension",
        "Expect 2D positions but got 3D positions",
    ],
)
def test_position_array_with_mismatched_dimensions(
    position_array, expected_error_message
):
    """Test validation for position_array dimension mismatches."""
    stub_dataset_class = make_stub_dataset_class()  # expects time, space
    with pytest.raises(
        ValueError,
        match=expected_error_message,
    ):
        stub_dataset_class(position_array=position_array)


@pytest.mark.parametrize(
    "expected_shape, expected_exception",
    [
        ((2, 3), does_not_raise()),
        (
            (3, 2),
            pytest.raises(
                ValueError,
                match=re.escape("have shape (3, 2), but got (2, 3)"),
            ),
        ),
    ],
)
def test_validate_array_shape(expected_shape, expected_exception):
    """Test the _validate_array_shape static method directly."""
    stub_dataset_class = make_stub_dataset_class()
    val = np.zeros((2, 3))
    with expected_exception:
        stub_dataset_class._validate_array_shape(
            StubAttr(), val, expected_shape
        )


@pytest.mark.parametrize(
    "val, expected_length, expected_exception",
    [
        (None, None, does_not_raise()),
        ([0, 1, 2], 3, does_not_raise()),
        (
            [0, 1],
            3,
            pytest.raises(
                ValueError,
                match="have length 3, but got 2",
            ),
        ),
    ],
)
def test_validate_list_length(val, expected_length, expected_exception):
    """Test the _validate_list_length method directly."""
    stub_dataset_class = make_stub_dataset_class()
    with expected_exception:
        stub_dataset_class._validate_list_length(
            StubAttr(), val, expected_length
        )


@pytest.mark.parametrize(
    "val, expected_exception",
    [
        (None, does_not_raise()),
        (["a", "b", "c"], does_not_raise()),
        (
            ["a", "b", "a"],
            pytest.raises(
                ValueError,
                match="are not unique",
            ),
        ),
    ],
)
def test_validate_list_uniqueness(val, expected_exception):
    """Test the _validate_list_uniqueness method directly."""
    stub_dataset_class = make_stub_dataset_class()
    with expected_exception:
        stub_dataset_class._validate_list_uniqueness(StubAttr(), val)


@pytest.mark.parametrize(
    "position_array, keypoint_names, expected_context",
    [
        (np.zeros((5, 2, 1, 2)), None, does_not_raise(["keypoint_0"])),
        (
            np.zeros((5, 2, 3, 2)),
            None,
            does_not_raise(["keypoint_0", "keypoint_1", "keypoint_2"]),
        ),
        (
            np.zeros((5, 2, 3, 2)),
            ["a", "b", "c"],
            does_not_raise(["a", "b", "c"]),
        ),
        (
            np.zeros((5, 2, 3, 2)),
            ["a", "b"],
            pytest.raises(ValueError, match="length 3, but got 2"),
        ),
    ],
    ids=[
        "Single keypoint default name",
        "Multiple keypoints default names",
        "Explicit keypoint names",
        "Explicit keypoint names length mismatch",
    ],
)
def test_poses_keypoint_names(
    position_array, keypoint_names, expected_context
):
    """Test keypoint_names validation in ValidPosesDataset."""
    with expected_context as expected_keypoint_names:
        ds = ValidPosesDataset(
            position_array=position_array,
            keypoint_names=keypoint_names,
        )
        assert ds.keypoint_names == expected_keypoint_names


@pytest.mark.parametrize(
    "source_software, position_array, expected_context",
    [
        ("LightningPose", np.zeros((5, 2, 3, 1)), does_not_raise()),
        (
            "LightningPose",
            np.zeros((5, 2, 3, 2)),
            pytest.raises(
                ValueError, match="LightningPose.*single-individual"
            ),
        ),
    ],
    ids=[
        "Valid single-individual LightningPose dataset",
        "Invalid multi-individual LightningPose dataset",
    ],
)
def test_poses_validate_position_array_impl(
    source_software, position_array, expected_context
):
    """Test additional position_array validation in ValidPosesDataset."""
    with expected_context:
        ValidPosesDataset(
            position_array=position_array,
            source_software=source_software,
        )
