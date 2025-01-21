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


@pytest.mark.parametrize(
    ["ref_vector", "expected_result"],
    [
        pytest.param(
            np.array([1.0, 0.0]),
            xr.DataArray(
                data=np.array([1.0, 0.0]),
                coords={"space": ["x", "y"]},
            ),
            id="(2,)-numpy array",
        ),
        pytest.param("x_axis", "x_axis", id="(2,) DataArray"),
        pytest.param(
            "x_axis_over_time", "x_axis_over_time", id="(n, 2) DataArray"
        ),
    ],
)
def test_validate_reference_vector(
    ref_vector: xr.DataArray | np.ndarray,
    expected_result: xr.DataArray | Exception,
    request,
) -> None:
    """Test that reference vectors or objects that can be cast to reference
    vectors, are cast correctly.

    For the purpose of testing this function, the only part of the
    `test_vector` that matters is the size of the `time` axis.
    As such, the `test_vector` will be assembled by calling
    `np.arange(0, 10, 1)`, and assigning this to the
    `time` axis.
    Therefore, it always has 10 entries, so acceptable reference vector
    shapes are thus
    - (2,)
    - (1, 2)
    - (10, 2)
    """
    if isinstance(ref_vector, str):
        ref_vector = request.getfixturevalue(ref_vector)
    if isinstance(expected_result, str):
        expected_result = request.getfixturevalue(expected_result)

    n_time_pts = 10
    test_vector = xr.DataArray(
        data=np.arange(0, n_time_pts, 1, dtype=float), dims="time"
    )

    if isinstance(expected_result, Exception):
        with pytest.raises(
            expected_result, match=re.escape(str(expected_result))
        ):
            validate_reference_vector(
                reference_vector=ref_vector, test_vector=test_vector
            )
    else:
        returned_vector = validate_reference_vector(
            reference_vector=ref_vector, test_vector=test_vector
        )

        xr.testing.assert_allclose(returned_vector, expected_result)
