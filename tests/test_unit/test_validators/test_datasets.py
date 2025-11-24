from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from movement.validators.datasets import (
    _BaseValidDataset,
    _convert_fps_to_none_if_invalid,
    _convert_to_list_of_str,
)


@pytest.mark.parametrize(
    "input, expected_context, expected_output",
    [
        (
            "abc",
            pytest.warns(
                UserWarning,
                match="Invalid value .* Expected a list of strings",
            ),
            ["abc"],
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


def make_stub_dataset_klass(
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


def test_required_fields_missing():
    """Test that TypeError is raised when required fields are missing."""
    StubDataset = make_stub_dataset_klass()
    with pytest.raises(
        TypeError,
        match="missing 1 required keyword-only argument: 'position_array'",
    ):
        StubDataset()


@pytest.mark.parametrize(
    "dim_names, position_array, expected_confidence, expected_ind_names",
    [
        (("time", "space"), np.random.rand(5, 2), np.full(5, np.nan), None),
        (
            ("time", "space", "individuals"),
            np.random.rand(5, 2, 1),
            np.full((5, 1), np.nan),
            ["id_0"],
        ),
        (
            ("time", "space", "individuals"),
            np.random.rand(5, 2, 3),
            np.full((5, 3), np.nan),
            ["id_0", "id_1", "id_2"],
        ),
    ],
)
def test_optional_fields_defaults(
    dim_names,
    position_array,
    expected_confidence,
    expected_ind_names,
):
    """Test default values for optional fields."""
    StubDataset = make_stub_dataset_klass(dim_names=dim_names)
    ds = StubDataset(position_array=position_array)
    np.testing.assert_allclose(
        ds.confidence_array, expected_confidence, equal_nan=True
    )
    assert ds.individual_names == expected_ind_names
    assert ds.fps is None
    assert ds.source_software is None


@pytest.mark.parametrize(
    "position_array, expected_error_message",
    [
        (np.random.rand(5, 2, 3), "have 2 dimensions, but got 3"),
        (np.random.rand(5, 3), "have 2 spatial dimensions, but got 3"),
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
    StubDataset = make_stub_dataset_klass()  # expects time, space
    with pytest.raises(
        ValueError,
        match=expected_error_message,
    ):
        StubDataset(position_array=position_array)


class StubAttr:
    """Minimal stub for attrs.Attribute."""

    def __init__(self, name="stub_attribute"):
        """Initialise with a name."""
        self.name = name


@pytest.mark.parametrize(
    "expected_shape, expected_exception",
    [
        ((2, 3), does_not_raise()),
        (
            (3, 2),
            pytest.raises(
                ValueError,
                match="have shape \(3, 2\), but got \(2, 3\)",
            ),
        ),
    ],
)
def test_validate_array_shape(expected_shape, expected_exception):
    """Test the _validate_array_shape static method directly."""
    StubDataset = make_stub_dataset_klass()
    val = np.zeros((2, 3))
    with expected_exception:
        StubDataset._validate_array_shape(StubAttr(), val, expected_shape)


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
    """Test the _validate_array_shape static method directly."""
    StubDataset = make_stub_dataset_klass()
    with expected_exception:
        StubDataset._validate_list_length(StubAttr(), val, expected_length)
