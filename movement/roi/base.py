"""Class for representing 2-dimensional regions of interest (RoIs)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import shapely

from movement.utils.logging import log_error

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

    - https://github.com/shapely/shapely/issues/1233#issuecomment-977837620.
    - https://stackoverflow.com/questions/10788976/how-do-i-properly-inherit-from-a-superclass-that-has-a-new-method

    To avoid the complexities of subclassing ourselves, we simply elect to wrap
    the appropriate ``shapely`` object in the ``_shapely_geometry`` attribute,
    accessible via the property ``region``. This also has the benefit of
    allowing us to 'forbid' certain operations (that ``shapely`` would
    otherwise interpret in a set-theoretic sense, giving confusing answers to
    users).

    This class is not designed to be instantiated directly. It _can be_
    instantiated, however its primary purpose is to reduce code duplication.
    """

    __default_name: str = "Un-named"
    __supported_type: TypeAlias = SupportedGeometry

    _name: str | None
    _shapely_geometry: __supported_type

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
        return not isinstance(self.region, shapely.LineString)

    @property
    def name(self) -> str:
        """Name of the instance."""
        return self._name if self._name else self.__default_name

    @property
    def region(self) -> __supported_type:
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
        """Initialise a region of interest."""
        self._name = name
        if len(points) < dimensions + 1:
            log_error(
                ValueError,
                f"Need at least {dimensions + 1} points to define a "
                f"{dimensions}D region (got {len(points)}).",
            )
        elif dimensions < 1 or dimensions > 2:
            log_error(
                ValueError,
                "Only regions of interest of dimension 1 or 2 are supported "
                f"(requested {dimensions})",
            )
        self._shapely_geometry = (
            shapely.Polygon(shell=points, holes=holes)
            if dimensions == 2
            else (
                shapely.LinearRing(coordinates=points)
                if closed
                else shapely.LineString(coordinates=points)
            )
        )

    def __repr__(self) -> str:  # noqa: D105
        return str(self)

    def __str__(self) -> str:  # noqa: D105
        display_type = "-gon" if self.dimensions > 1 else "line segment(s)"
        return (
            f"{self.__class__.__name__} {self.name} "
            f"({len(self.region.coords)}{display_type})\n"
            " -> ".join(f"({c[0]}, {c[1]})" for c in self.region.coords)
        )
