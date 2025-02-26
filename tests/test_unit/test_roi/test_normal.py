import re

import numpy as np
import pytest
from numpy.typing import ArrayLike

from movement.roi import LineOfInterest

SQRT_2 = np.sqrt(2.0)


@pytest.mark.parametrize(
    ["segment", "point", "expected_normal"],
    [
        pytest.param(
            "segment_of_y_equals_x",
            (0.0, 1.0),
            (-1.0 / SQRT_2, 1.0 / SQRT_2),
            id="Normal pointing to half-plane with point (0, 1)",
        ),
        pytest.param(
            "segment_of_y_equals_x",
            (1.0, 0.0),
            (1.0 / SQRT_2, -1.0 / SQRT_2),
            id="Normal pointing to half-plane with point (1, 0)",
        ),
        pytest.param(
            LineOfInterest([(0.5, 0.5), (1.0, 1.0)]),
            (1.0, 0.0),
            (1.0 / SQRT_2, -1.0 / SQRT_2),
            id="Segment does not start at origin",
        ),
        pytest.param(
            "segment_of_y_equals_x",
            (1.0, 2.0),
            (-1.0 / SQRT_2, 1.0 / SQRT_2),
            id="Necessary to extend segment to compute normal.",
        ),
        pytest.param(
            LineOfInterest([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]),
            (0.5, 0.5),
            ValueError("Normal is not defined for multi-segment geometries."),
            id="Multi-segment lines do not have normals.",
        ),
    ],
)
def test_normal(
    segment: LineOfInterest,
    point: ArrayLike,
    expected_normal: np.ndarray | Exception,
    request,
) -> None:
    if isinstance(segment, str):
        segment = request.getfixturevalue(segment)

    if isinstance(expected_normal, Exception):
        with pytest.raises(
            type(expected_normal), match=re.escape(str(expected_normal))
        ):
            segment.normal(point)
    else:
        computed_normal = segment.normal(point)
        assert np.allclose(computed_normal, expected_normal)
