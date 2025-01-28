from collections.abc import Callable
from typing import Any, Concatenate

import numpy as np
import pytest
import xarray as xr

from movement.utils.broadcasting import (
    KeywordArgs,
    Scalar,
    make_broadcastable,
    make_broadcastable_over_space,
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
    fn_args: Any,
    fn_kwargs: Any,
) -> None:
    if isinstance(expected_output, np.ndarray):
        reduced_dims = list(mock_dataset.dims)
        reduced_dims.remove(along_dimension)
        reduced_coords = dict(mock_dataset.coords)
        reduced_coords.pop(along_dimension, None)

        expected_output = xr.DataArray(
            data=expected_output, dims=reduced_dims, coords=reduced_coords
        )
    decorated_fn = make_broadcastable(mimic_fn)

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
        decorated_fn_space_only = make_broadcastable_over_space(mimic_fn)
        decorated_output_space = decorated_fn_space_only(
            mock_dataset, *fn_args, **fn_kwargs
        )

        assert decorated_output_space.shape == expected_output.shape
        xr.testing.assert_allclose(decorated_output_space, expected_output)
