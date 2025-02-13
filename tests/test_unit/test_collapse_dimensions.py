import re
from typing import Any

import numpy as np
import pytest
import xarray as xr

from movement.utils.dimensions import (
    _coord_of_dimension,
    collapse_extra_dimensions,
)


@pytest.fixture
def shape() -> tuple[int, ...]:
    return (7, 2, 4, 3)


@pytest.fixture
def da(shape: tuple[int, ...]) -> xr.DataArray:
    return xr.DataArray(
        data=np.arange(np.prod(shape)).reshape(shape),
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "space": ["x", "y"],
            "keypoints": ["head", "shoulders", "knees", "toes"],
            "individuals": ["a", "b", "c"],
        },
    )


@pytest.mark.parametrize(
    ["pass_to_function", "equivalent_to_sel"],
    [
        pytest.param(
            {},
            {"individuals": "a", "keypoints": "head"},
            id="Default preserve time-space",
        ),
        pytest.param(
            {"preserve_dims": ["space"]},
            {"time": 0, "individuals": "a", "keypoints": "head"},
            id="Keep space only",
        ),
        pytest.param(
            {"individuals": "b"},
            {"individuals": "b", "keypoints": "head"},
            id="Request non-default slice",
        ),
        pytest.param(
            {"individuals": "c"},
            {"individuals": "c", "keypoints": "head"},
            id="Request by coordinate",
        ),
        pytest.param(
            {
                "individuals": "b",
                "elephants": "this is a non-existent dimension",
                "crabs": 42,
            },
            {"individuals": "b", "keypoints": "head"},
            id="Selection ignores dimensions that don't exist",
        ),
        pytest.param(
            {"preserve_dims": []},
            {"time": 0, "space": "x", "individuals": "a", "keypoints": "head"},
            id="Collapse everything",
        ),
    ],
)
def test_collapse_dimensions(
    da: xr.DataArray,
    pass_to_function: dict[str, str],
    equivalent_to_sel: dict[str, int | str],
) -> None:
    result_from_collapsing = collapse_extra_dimensions(da, **pass_to_function)

    # We should be equivalent to this method
    expected_result = da.sel(**equivalent_to_sel)

    assert result_from_collapsing.shape == expected_result.values.shape
    xr.testing.assert_allclose(result_from_collapsing, expected_result)


@pytest.mark.parametrize(
    ["pass_to_function"],
    [
        pytest.param(
            {"keypoints": ["head", "toes"]},
            id="Multiple keypoints",
        ),
        pytest.param(
            {"individuals": ["a", "b"]},
            id="Multiple individuals",
        ),
    ],
)
def test_collapse_dimensions_value_error(
    da: xr.DataArray,
    pass_to_function: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        collapse_extra_dimensions(da, **pass_to_function)


@pytest.mark.parametrize(
    ["args_to_fn", "expected"],
    [
        pytest.param(
            {"dimension": "individuals", "coord_index": 1},
            "b",
            id="Fetch coord from index",
        ),
        pytest.param(
            {"dimension": "time", "coord_index": 6},
            6,
            id="Dimension with no coordinates",
        ),
        pytest.param(
            {"dimension": "space", "coord_index": "x"},
            "x",
            id="Fetch coord from name",
        ),
        pytest.param(
            {"dimension": "keypoints", "coord_index": 10},
            IndexError("index 10 is out of bounds for axis 0 with size 4"),
            id="Out of bounds index",
        ),
        pytest.param(
            {"dimension": "keypoints", "coord_index": "arms"},
            KeyError("arms"),
            id="Non existent coord name",
        ),
    ],
)
def test_coord_of_dimension(
    da: xr.DataArray, args_to_fn: dict[str, str], expected: str | Exception
) -> None:
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=re.escape(str(expected))):
            _coord_of_dimension(da, **args_to_fn)
    else:
        assert expected == _coord_of_dimension(da, **args_to_fn)
