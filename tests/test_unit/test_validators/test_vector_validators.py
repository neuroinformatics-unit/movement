from __future__ import annotations

import re

import numpy as np
import pytest
import xarray as xr

from movement.validators.vector import validate_reference_vector


@pytest.fixture
def x_axis() -> xr.DataArray:
    return xr.DataArray(
        data=np.array([1.0, 0.0]),
        coords={"space": ["x", "y"]},
    )


def x_axis_over_time(n_time_pts: int = 10) -> xr.DataArray:
    """Return the x-axis as a constant DataArray,
    with as many time points as requested.
    """
    return xr.DataArray(
        data=np.tile(np.array([1.0, 0.0]).reshape(1, -1), [n_time_pts, 1]),
        dims=["time", "space"],
        coords={"time": np.arange(n_time_pts), "space": ["x", "y"]},
    )


@pytest.mark.parametrize(
    ["ref_vector", "test_vector", "expected_result"],
    [
        pytest.param(
            np.array([1.0, 0.0]),
            x_axis_over_time(),
            "x_axis",
            id="(2,) numpy array vs (n, 2) DataArray",
        ),
        pytest.param(
            np.array([1.0, 0.0]),
            "x_axis",
            "x_axis",
            id="(2,) numpy array vs (1, 2) DataArray",
        ),
        pytest.param(
            np.array([1.0, 0.0]).reshape(1, -1),
            x_axis_over_time(),
            "x_axis",
            id="(1, 2) numpy array vs (n, 2) DataArray",
        ),
        pytest.param(
            "x_axis",
            x_axis_over_time(),
            "x_axis",
            id="(1, 2) DataArray vs (n, 2) DataArray",
        ),
        pytest.param(
            x_axis_over_time(),
            x_axis_over_time(),
            x_axis_over_time(),
            id="(n, 2) DataArray vs (n, 2) DataArray",
        ),
        pytest.param(
            np.tile(np.array([1.0, 0.0]).reshape(1, -1), [10, 1]),
            x_axis_over_time(),
            x_axis_over_time(),
            id="(n, 2) numpy array vs (n, 2) DataArray",
        ),
        pytest.param(
            x_axis_over_time(5),
            x_axis_over_time(),
            ValueError(
                "Reference vector must have the same number "
                "of time points as the test vector."
            ),
            id="Too few time points (numpy)",
        ),
        pytest.param(
            np.tile(np.array([1.0, 0.0]).reshape(1, -1), [5, 1]),
            x_axis_over_time(),
            ValueError(
                "Reference vector must have the same number "
                "of time points as the test vector."
            ),
            id="Too few time points (DataArray)",
        ),
        pytest.param(
            np.ones(shape=(2, 2, 2)),
            x_axis_over_time(),
            ValueError("Reference vector must be 1D or 2D, but got 3D array."),
            id="Too many dimensions",
        ),
        pytest.param(
            xr.DataArray(
                data=np.ones(shape=(10, 2, 1)),
                dims=["time", "space", "elephants"],
                coords={
                    "time": np.arange(10),
                    "space": ["x", "y"],
                    "elephants": ["e"],
                },
            ),
            x_axis_over_time(10),
            ValueError(
                "Only 'time' and 'space' dimensions "
                "are allowed in reference_vector.",
            ),
            id="Extra dimension in reference vector",
        ),
        pytest.param(
            np.pi,
            "x_axis",
            TypeError(
                "Reference vector must be an xarray.DataArray or np.ndarray, "
                "but got <class 'float'>."
            ),
            id="Wrong input type",
        ),
    ],
)
def test_validate_reference_vector(
    ref_vector: xr.DataArray | np.ndarray,
    test_vector: xr.DataArray,
    expected_result: xr.DataArray | Exception,
    request,
) -> None:
    """Test that reference vectors or objects that can be cast to reference
    vectors, are cast correctly.
    """
    if isinstance(ref_vector, str):
        ref_vector = request.getfixturevalue(ref_vector)
    if isinstance(test_vector, str):
        test_vector = request.getfixturevalue(test_vector)
    if isinstance(expected_result, str):
        expected_result = request.getfixturevalue(expected_result)

    if isinstance(expected_result, Exception):
        with pytest.raises(
            type(expected_result), match=re.escape(str(expected_result))
        ):
            validate_reference_vector(
                reference_vector=ref_vector, test_vector=test_vector
            )
    else:
        returned_vector = validate_reference_vector(
            reference_vector=ref_vector, test_vector=test_vector
        )

        xr.testing.assert_allclose(returned_vector, expected_result)
