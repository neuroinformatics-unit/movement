import numpy as np
import pytest

from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest
from movement.transforms import compute_homography_transform


@pytest.mark.parametrize(
    ["coords_a", "coords_b"],
    [
        pytest.param(
            np.array([[1, 1], [5, 1], [5, 3], [1, 3]], dtype=np.float32),
            np.array([[1, 1], [5, 1], [5, 3], [1, 3]], dtype=np.float32),
            id="Identical rectangles (identity transform)",
        ),
        pytest.param(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            np.array(
                [
                    [3, -1],
                    [4.73205081, 0],
                    [3.98205081, 1.29903811],
                    [2.25, 0.2990381],
                ],
                dtype=np.float32,
            ),
            id="Rotated and scaled square",
        ),
    ],
)
def test_get_transform_happy_path(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> None:
    roi_a = PolygonOfInterest(coords_a)
    roi_b = PolygonOfInterest(coords_b)

    expected_transform = compute_homography_transform(coords_a, coords_b)

    computed_transform = roi_a.get_transform(roi_b)

    assert computed_transform.shape == (3, 3)
    assert np.allclose(computed_transform, expected_transform, atol=1e-6)


@pytest.mark.parametrize(
    ["coords_a", "coords_b"],
    [
        pytest.param(
            np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32),
            np.array(
                [[0, 0], [1, 0], [2, 0.5], [1, 1], [0, 1]], dtype=np.float32
            ),
            id="Triangle vs Pentagon",
        ),
        pytest.param(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32),
            id="Quad vs triangle",
        ),
    ],
)
def test_get_transform_mismatched_points_raises(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> None:
    roi_a = PolygonOfInterest(coords_a)
    roi_b = PolygonOfInterest(coords_b)

    with pytest.raises(ValueError):
        roi_a.get_transform(roi_b)


@pytest.mark.parametrize(
    ["coords_a", "coords_b"],
    [
        pytest.param(
            np.array([[0, 0], [1, 1]], dtype=np.float32),
            np.array([[0, 0], [1, 2]], dtype=np.float32),
            id="2 lines",
        )
    ],
)
def test_line_of_interest_raises(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
):
    line_1 = LineOfInterest(points=coords_a)
    line_2 = LineOfInterest(points=coords_b)

    with pytest.raises(NotImplementedError):
        line_1.get_transform(line_2)
