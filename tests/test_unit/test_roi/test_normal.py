import numpy as np
import pytest
from numpy.typing import ArrayLike

from movement.roi import LineOfInterest

SQRT_2 = np.sqrt(2.0)


@pytest.mark.parametrize(
    ["point", "expected_normal"],
    [
        pytest.param(
            (0.0, 1.0), (-1.0 / SQRT_2, 1.0 / SQRT_2), id="Normal to (0, 1)"
        ),
        pytest.param(
            (1.0, 0.0), (1.0 / SQRT_2, -1.0 / SQRT_2), id="Normal to (1, 0)"
        ),
        pytest.param(
            (1.0, 2.0), (-1.0 / SQRT_2, 1.0 / SQRT_2), id="Normal to (1, 2)"
        ),
    ],
)
def test_normal(
    segment_of_y_equals_x: LineOfInterest,
    point: ArrayLike,
    expected_normal: np.ndarray,
) -> None:
    computed_normal = segment_of_y_equals_x.normal(point)

    assert np.allclose(computed_normal, expected_normal)
