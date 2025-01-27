import numpy as np
import pytest


@pytest.fixture()
def unit_square_pts() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )


@pytest.fixture()
def unit_square_hole(unit_square_pts: np.ndarray) -> np.ndarray:
    """Hole in the shape of a 0.25 side-length square centred on 0.5, 0.5."""
    return 0.25 + (unit_square_pts.copy() * 0.5)
