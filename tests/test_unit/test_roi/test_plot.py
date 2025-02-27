from typing import Any

import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch

from movement.roi import LineOfInterest, PolygonOfInterest
from movement.roi.base import BaseRegionOfInterest


@pytest.fixture
def havarti() -> PolygonOfInterest:
    """Wedge-shaped RoI with several holes.

    Havarti is a type of cheese that typically has holes and divots.
    """
    wedge = [(0, 0), (4, 0), (4, 0.5), (0, 2)]
    hole1 = [(0.5, 0.5), (0.5, 0.75), (0.625, 0.625)]
    hole2 = [(3, 0.25), (3.25, 0.5), (3.25, 0.75), (2.75, 0.75), (2.625, 0.5)]
    hole3 = [(1.0, 0.1), (2, 0.1), (1.75, 0.2), (1.5, 0.3), (1.25, 0.2)]
    hole4 = [(0.5, 1.75), (0.75, 1.5), (1.0, 1.25), (1.25, 1.25), (1.5, 1.4)]
    return PolygonOfInterest(
        wedge, holes=[hole1, hole2, hole3, hole4], name="Havarti"
    )


@pytest.fixture
def octagonal_doughnut() -> PolygonOfInterest:
    """18-sided doughnut.

    This region matches (approximately) to the arena in the
    "SLEAP_three-mice_Aeon_proofread.analysis.h5" dataset.
    """
    centre = np.array([712.5, 541])
    width = 40.0
    extent = 1090.0

    n_pts = 18
    unit_shape = np.array(
        [
            np.exp((np.pi / 2.0 + (2.0 * i * np.pi) / n_pts) * 1.0j)
            for i in range(n_pts)
        ],
        dtype=complex,
    )
    outer_boundary = extent / 2.0 * unit_shape.copy()
    outer_boundary = (
        np.array([outer_boundary.real, outer_boundary.imag]).transpose()
        + centre
    )
    inner_boundary = (extent - width) / 2.0 * unit_shape.copy()
    inner_boundary = (
        np.array([inner_boundary.real, inner_boundary.imag]).transpose()
        + centre
    )
    return PolygonOfInterest(
        outer_boundary, holes=[inner_boundary], name="Arena"
    )


@pytest.mark.parametrize(
    ["region_to_plot", "kwargs"],
    [
        pytest.param("unit_square", {}, id="Unit square"),
        pytest.param("unit_square_with_hole", {}, id="Unit square with hole"),
        pytest.param(
            "havarti",
            {"facecolor": "yellow", "edgecolor": "black"},
            id="Cheese",
        ),
        pytest.param(
            "octagonal_doughnut",
            {"facecolor": ("black", 0.0)},  # Transparency hack
            id="Octagonal doughnut",
        ),
        pytest.param(
            LineOfInterest([(0.0, 0.0), (1.0, 0.0)]), {}, id="Segment"
        ),
        pytest.param(
            LineOfInterest([(0.0, 0.0), (1.0, 0), (1.0, 1.0), (0.0, 1)]),
            {},
            id="Multi-segment",
        ),
    ],
)
def test_plot(
    region_to_plot: BaseRegionOfInterest,
    kwargs: dict[str, Any],
    request,
) -> None:
    if isinstance(region_to_plot, str):
        region_to_plot = request.getfixturevalue(region_to_plot)

    _, ax = region_to_plot.plot(**kwargs)
    if region_to_plot.dimensions == 2:
        assert len(ax.patches) == 1 and len(ax.lines) == 0
        assert type(ax.patches[0]) is PathPatch
    else:
        assert len(ax.patches) == 0 and len(ax.lines) == 1
        assert type(ax.lines[0]) is Line2D
