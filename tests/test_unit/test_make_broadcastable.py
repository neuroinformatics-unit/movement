from collections.abc import Callable
from typing import Any, Concatenate

import numpy as np
import pytest
import xarray as xr

from movement.utils.broadcasting import (
    KeywordArgs,
    ScalarOr1D,
    broadcastable_method,
    make_broadcastable,
    space_broadcastable,
)


def copy_with_collapsed_dimension(
    original: xr.DataArray, collapse: str, new_data: np.ndarray
) -> xr.DataArray:
    reduced_dims = list(original.dims)
    reduced_dims.remove(collapse)
    reduced_coords = dict(original.coords)
    reduced_coords.pop(collapse, None)

    return xr.DataArray(
        data=new_data, dims=reduced_dims, coords=reduced_coords
    )


def data_in_shape(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(np.prod(shape), dtype=float).reshape(shape)


def mock_shape() -> tuple[int, ...]:
    return (10, 2, 4, 3)


@pytest.fixture
def mock_data_array() -> xr.DataArray:
    return xr.DataArray(
        data=data_in_shape(mock_shape()),
        dims=["time", "space", "keypoints", "individuals"],
        coords={"space": ["x", "y"]},
    )


@pytest.mark.parametrize(
    ["along_dimension", "expected_output", "mimic_fn", "fn_args", "fn_kwargs"],
    [
        pytest.param(
            "space",
            np.zeros(mock_shape()).sum(axis=1),
            lambda x: 0.0,
            tuple(),
            {},
            id="Zero everything",
        ),
        pytest.param(
            "space",
            data_in_shape(mock_shape()).sum(axis=1),
            sum,
            tuple(),
            {},
            id="Mimic sum",
        ),
        pytest.param(
            "time",
            data_in_shape(mock_shape()).prod(axis=0),
            np.prod,
            tuple(),
            {},
            id="Mimic prod, on non-space dimensions",
        ),
        pytest.param(
            "space",
            5.0 * data_in_shape(mock_shape()).sum(axis=1),
            lambda x, **kwargs: kwargs.get("multiplier", 1.0) * sum(x),
            tuple(),
            {"multiplier": 5.0},
            id="Preserve kwargs",
        ),
        pytest.param(
            "space",
            data_in_shape(mock_shape()).sum(axis=1),
            lambda x, **kwargs: kwargs.get("multiplier", 1.0) * sum(x),
            tuple(),
            {},
            id="Preserve kwargs [fall back on default]",
        ),
        pytest.param(
            "space",
            5.0 * data_in_shape(mock_shape()).sum(axis=1),
            lambda x, multiplier=1.0: multiplier * sum(x),
            (5,),
            {},
            id="Preserve args",
        ),
        pytest.param(
            "space",
            data_in_shape(mock_shape()).sum(axis=1),
            lambda x, multiplier=1.0: multiplier * sum(x),
            tuple(),
            {},
            id="Preserve args [fall back on default]",
        ),
    ],
)
def test_make_broadcastable(
    mock_data_array: xr.DataArray,
    along_dimension: str,
    expected_output: xr.DataArray,
    mimic_fn: Callable[Concatenate[Any, KeywordArgs], ScalarOr1D],
    fn_args: list[Any],
    fn_kwargs: dict[str, Any],
) -> None:
    """Test make_broadcastable decorator, when acting on functions."""
    if isinstance(expected_output, np.ndarray):
        expected_output = copy_with_collapsed_dimension(
            mock_data_array, along_dimension, expected_output
        )
    decorated_fn = make_broadcastable()(mimic_fn)

    decorated_output = decorated_fn(
        mock_data_array,
        *fn_args,
        broadcast_dimension=along_dimension,  # type: ignore
        **fn_kwargs,
    )

    assert decorated_output.shape == expected_output.shape
    xr.testing.assert_allclose(decorated_output, expected_output)

    # Also check the case where we only want to be able to cast over space.
    if along_dimension == "space":
        decorated_fn_space_only = space_broadcastable()(mimic_fn)
        decorated_output_space = decorated_fn_space_only(
            mock_data_array, *fn_args, **fn_kwargs
        )

        assert decorated_output_space.shape == expected_output.shape
        xr.testing.assert_allclose(decorated_output_space, expected_output)


@pytest.mark.parametrize(
    [
        "along_dimension",
        "cls_attribute",
        "fn_args",
        "fn_kwargs",
        "expected_output",
    ],
    [
        pytest.param(
            "space",
            1.0,
            [1.0],
            {},
            data_in_shape(mock_shape()).sum(axis=1) + 1.0,
            id="In space",
        ),
        pytest.param(
            "time",
            5.0,
            [],
            {"c": 2.5},
            5.0 * data_in_shape(mock_shape()).sum(axis=0) + 2.5,
            id="In time",
        ),
    ],
)
def test_make_broadcastable_classmethod(
    mock_data_array: xr.DataArray,
    along_dimension: str,
    cls_attribute: float,
    fn_args: list[Any],
    fn_kwargs: dict[str, Any],
    expected_output: np.ndarray,
) -> None:
    """Test make_broadcastable decorator, when acting on class methods."""

    class DummyClass:
        mult: float

        def __init__(self, multiplier=1.0):
            self.mult = multiplier

        @broadcastable_method()
        def sum_and_mult_plus_c(self, values, c):
            return self.mult * sum(values) + c

    expected_output = copy_with_collapsed_dimension(
        mock_data_array, along_dimension, expected_output
    )
    d = DummyClass(cls_attribute)

    decorated_output = d.sum_and_mult_plus_c(
        mock_data_array,
        *fn_args,
        broadcast_dimension=along_dimension,
        **fn_kwargs,
    )

    assert decorated_output.shape == expected_output.shape
    xr.testing.assert_allclose(decorated_output, expected_output)


@pytest.mark.parametrize(
    ["broadcast_dim", "new_dim_length", "new_dim_name"],
    [
        pytest.param("space", 3, None, id="(3,), default dim name"),
        pytest.param("space", 5, "elephants", id="(5,), custom dim name"),
    ],
)
def test_vector_outputs(
    mock_data_array: xr.DataArray,
    broadcast_dim: str,
    new_dim_length: int,
    new_dim_name: str | None,
) -> None:
    """Test make_broadcastable when 1D vector outputs are provided."""
    if not new_dim_name:
        # Take on the default value given in the method,
        # if not provided
        new_dim_name = "result"

    @make_broadcastable(
        only_broadcastable_along=broadcast_dim, new_dimension_name=new_dim_name
    )
    def two_to_some(xy_pair) -> np.ndarray:
        # A 1D -> 1D function, rather than a function that returns a scalar.
        return np.linspace(
            xy_pair[0], xy_pair[1], num=new_dim_length, endpoint=True
        )

    output = two_to_some(mock_data_array)

    assert isinstance(output, xr.DataArray)
    for d in output.dims:
        if d == new_dim_name:
            assert len(output[d]) == new_dim_length
        else:
            assert d in mock_data_array.dims
            assert len(output[d]) == len(mock_data_array[d])


def test_retain_underlying_function() -> None:
    value_for_arg = 5.0
    value_for_kwarg = 7.0
    value_for_simple_input = [0.0, 1.0, 2.0]

    def simple_function(input_1d, arg, kwarg=3.0):
        return arg * sum(input_1d) + kwarg

    @make_broadcastable()
    def simple_function_broadcastable(input_1d, arg, kwarg=3.0):
        return simple_function(input_1d, arg, kwarg=kwarg)

    class DummyClass:
        factor: float

        def __init__(self, factor: float = 1.0):
            self.factor = factor

        @broadcastable_method(only_broadcastable_along="space")
        def simple_broadcastable_method(self, values, kwarg=3.0) -> float:
            return simple_function(values, self.factor, kwarg=kwarg)

    result_from_original = simple_function(
        value_for_simple_input, value_for_arg, kwarg=value_for_kwarg
    )
    result_from_broadcastable = simple_function_broadcastable(
        value_for_simple_input, value_for_arg, kwarg=value_for_kwarg
    )
    result_from_clsmethod = DummyClass(
        value_for_arg
    ).simple_broadcastable_method(
        value_for_simple_input, kwarg=value_for_kwarg
    )

    assert isinstance(result_from_broadcastable, float)
    assert isinstance(result_from_clsmethod, float)

    assert np.isclose(result_from_broadcastable, result_from_original)
    assert np.isclose(result_from_clsmethod, result_from_original)
