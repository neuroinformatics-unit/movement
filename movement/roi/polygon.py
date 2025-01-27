"""2-dimensional regions of interest."""

from collections.abc import Sequence
from typing import TypeAlias

from movement.roi.base import BaseRegionOfInterest, PointLikeList, RegionLike
from movement.roi.line import LineOfInterest


class PolygonOfInterest(BaseRegionOfInterest):
    """Representation of a two-dimensional region in the x-y plane.

    This class can be used to represent polygonal regions or subregions
    of the area in which the experimental data was gathered. These might
    include the arms of a maze, a nesting area, a food source, or other
    similar areas of the experimental enclosure that have some significance.

    An instance of this class can be used to represent these regions of
    interest (RoIs) in an analysis. The basic usage is to construct an
    instance of this class by passing in a list of points, which will then be
    joined (in sequence) by straight lines between consecutive pairs of points,
    to form the boundary of the RoI. Note that the boundary itself is a
    (closed) ``LineOfInterest``, and may be treated accordingly.
    """

    __default_name: str = "Un-named polygon"
    __supported_type: TypeAlias = RegionLike

    def __init__(
        self,
        boundary: PointLikeList,
        holes: Sequence[PointLikeList] | None = None,
        name: str | None = None,
    ) -> None:
        """Create a new region of interest (RoI).

        Parameters
        ----------
        boundary : tuple of (x, y) pairs
            The points (in sequence) that make up the boundary of the region.
            At least three points must be provided.
        holes : sequence of tuples of (x, y) pairs
            A sequence of items that will be interpreted like ``boundary``,
            that will be used to construct internal holes within the region.
            See the ``holes`` argument to ``shapely.Polygon`` for details.
        name : str
            Name of the RoI that is to be created.

        """
        super().__init__(points=boundary, dimensions=2, holes=holes, name=name)

    @property
    def boundary(self) -> LineOfInterest:
        """The boundary of this RoI."""
        return LineOfInterest(
            self.region.boundary.coords,
            loop=True,
            name=f"Boundary of {self.name}",
        )
