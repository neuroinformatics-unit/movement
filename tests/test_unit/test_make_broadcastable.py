from collections.abc import Callable
from typing import Any, Concatenate

import numpy as np
import pytest
import xarray as xr

from movement.utils.broadcasting import (
    KeywordArgs,
    Scalar,
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
    return (10, 2, 3, 4)


@pytest.fixture
def mock_dataset() -> xr.DataArray:
    return xr.DataArray(
        data=data_in_shape(mock_shape()),
        dims=["time", "space", "individuals", "keypoints"],
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
    mock_dataset: xr.DataArray,
    along_dimension: str,
    expected_output: xr.DataArray,
    mimic_fn: Callable[Concatenate[Any, KeywordArgs], Scalar],
    fn_args: list[Any],
    fn_kwargs: dict[str, Any],
) -> None:
    """Test make_broadcastable decorator, when acting on functions."""
    if isinstance(expected_output, np.ndarray):
        expected_output = copy_with_collapsed_dimension(
            mock_dataset, along_dimension, expected_output
        )
    decorated_fn = make_broadcastable()(mimic_fn)

    decorated_output = decorated_fn(
        mock_dataset,
        *fn_args,
        broadcast_dimension=along_dimension,  # type: ignore
        **fn_kwargs,
    )

    assert decorated_output.shape == expected_output.shape
    xr.testing.assert_allclose(decorated_output, expected_output)

    # Also check the case where we only want to be able to cast over time.
    if along_dimension == "space":
        decorated_fn_space_only = space_broadcastable()(mimic_fn)
        decorated_output_space = decorated_fn_space_only(
            mock_dataset, *fn_args, **fn_kwargs
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
    mock_dataset: xr.DataArray,
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
        mock_dataset, along_dimension, expected_output
    )
    d = DummyClass(cls_attribute)

    decorated_output = d.sum_and_mult_plus_c(
        mock_dataset,
        *fn_args,
        broadcast_dimension=along_dimension,
        **fn_kwargs,
    )

    assert decorated_output.shape == expected_output.shape
    xr.testing.assert_allclose(decorated_output, expected_output)
