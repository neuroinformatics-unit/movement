"""Class for representing 2-dimensional regions of interest (RoIs)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

import shapely
from shapely.coords import CoordinateSequence

from movement.utils.broadcasting import broadcastable_method
from movement.utils.logging import log_error

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

LineLike: TypeAlias = shapely.LinearRing | shapely.LineString
PointLike: TypeAlias = tuple[float, float]
PointLikeList: TypeAlias = Sequence[PointLike]
RegionLike: TypeAlias = shapely.Polygon
SupportedGeometry: TypeAlias = LineLike | RegionLike


class BaseRegionOfInterest:
    """Base class for representing regions of interest (RoI)s.

    Regions of interest can be either 1 or 2 dimensional, and are represented
    by appropriate ``shapely.Geometry`` objects depending on which. Note that
    there are a number of discussions concerning subclassing ``shapely``
    objects;

    - https://github.com/shapely/shapely/issues/1233.
    - https://stackoverflow.com/questions/10788976/how-do-i-properly-inherit-from-a-superclass-that-has-a-new-method

    To avoid the complexities of subclassing ourselves, we simply elect to wrap
    the appropriate ``shapely`` object in the ``_shapely_geometry`` attribute,
    accessible via the property ``region``. This also has the benefit of
    allowing us to 'forbid' certain operations (that ``shapely`` would
    otherwise interpret in a set-theoretic sense, giving confusing answers to
    users).

    This class is not designed to be instantiated directly. It can be
    instantiated, however its primary purpose is to reduce code duplication.
    """

    __default_name: str = "Unnamed region"

    _name: str | None

    @property
    def coords(self) -> CoordinateSequence:
        """Coordinates of the points that define the region.

        These are the points passed to the constructor argument ``points``.

        Note that for Polygonal regions, these are the coordinates of the
        exterior boundary, interior boundaries must be accessed via
        ``self.region.interior.coords``.
        """
        return (
            self.region.coords
            if self.dimensions < 2
            else self.region.exterior.coords
        )

    @property
    def dimensions(self) -> int:
        """Dimensionality of the region."""
        return shapely.get_dimensions(self.region)

    @property
    def is_closed(self) -> bool:
        """Return True if the region is closed.

        A closed region is either:
        - A polygon (2D RoI).
        - A 1D LoI whose final point connects back to its first.
        """
        return self.dimensions > 1 or (
            self.dimensions == 1
            and self.region.coords[0] == self.region.coords[-1]
        )

    @property
    def name(self) -> str:
        """Name of the instance."""
        return self._name if self._name else self.__default_name

    @property
    def region(self) -> SupportedGeometry:
        """``shapely.Geometry`` representation of the region."""
        return self._shapely_geometry

    def __init__(
        self,
        points: PointLikeList,
        dimensions: Literal[1, 2] = 2,
        closed: bool = False,
        holes: Sequence[PointLikeList] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialise a region of interest.

        Parameters
        ----------
        points : Sequence of (x, y) values
            Sequence of (x, y) coordinate pairs that will form the region.
        dimensions : Literal[1, 2], default 2
            The dimensionality of the region to construct.
            '1' creates a sequence of joined line segments,
            '2' creates a polygon whose boundary is defined by ``points``.
        closed : bool, default False
            Whether the line to be created should be closed. That is, whether
            the final point should also link to the first point.
            Ignored if ``dimensions`` is 2.
        holes : sequence of sequence of (x, y) pairs, default None
            A sequence of items, where each item will be interpreted like
            ``points``. These items will be used to construct internal holes
            within the region. See the ``holes`` argument to
            ``shapely.Polygon`` for details. Ignored if ``dimensions`` is 1.
        name : str, default None
            Human-readable name to assign to the given region, for
            user-friendliness.

        """
        self._name = name
        if len(points) < dimensions + 1:
            raise log_error(
                ValueError,
                f"Need at least {dimensions + 1} points to define a "
                f"{dimensions}D region (got {len(points)}).",
            )
        elif dimensions < 1 or dimensions > 2:
            raise log_error(
                ValueError,
                "Only regions of interest of dimension 1 or 2 are supported "
                f"(requested {dimensions})",
            )
        elif dimensions == 1 and len(points) < 3 and closed:
            raise log_error(
                ValueError,
                "Cannot create a loop from a single line segment.",
            )

        # Assign underlying geometry
        if dimensions == 2:
            self._shapely_geometry = shapely.Polygon(shell=points, holes=holes)
        else:
            self._shapely_geometry = (
                shapely.LinearRing(coordinates=points)
                if closed
                else shapely.LineString(coordinates=points)
            )

    def __repr__(self) -> str:  # noqa: D105
        return str(self)

    def __str__(self) -> str:  # noqa: D105
        display_type = "-gon" if self.dimensions > 1 else " line segment(s)"
        n_points = len(self.coords) - 1
        return (
            f"{self.__class__.__name__} {self.name} "
            f"({n_points}{display_type})\n"
        ) + " -> ".join(f"({c[0]}, {c[1]})" for c in self.coords)

    @broadcastable_method(only_broadcastable_along="space")
    def points_are_inside(
        self,
        /,
        position: ArrayLike,
        include_boundary: bool = True,
    ) -> bool:
        """Determine if a position is inside the region of interest.

        Parameters
        ----------
        position : ArrayLike
            Spatial coordinates [x, y, [z]] to check as being inside the
            region.
        include_boundary : bool
            Whether to treat a position on the region's boundary as inside the
            region (True) or outside the region (False). Default True.

        Returns
        -------
        bool
            True if the ``position`` provided is within the region of interest.
            False otherwise.

        """
        point = shapely.Point(position)

        current_region = self.region
        point_is_inside = current_region.contains(point)

        if include_boundary:
            # 2D objects have 1D object boundaries,
            # which in turn have point-boundaries.
            while not current_region.boundary.is_empty:
                current_region = current_region.boundary
                point_is_inside = point_is_inside or current_region.contains(
                    point
                )
        return point_is_inside
