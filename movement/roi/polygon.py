"""2-dimensional regions of interest."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch as PltPatch
from matplotlib.path import Path as PltPath

from movement.roi.base import BaseRegionOfInterest, PointLikeList
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
    to form the exterior boundary of the RoI. Note that the exterior boundary
    (accessible as via the ``.exterior`` property) is a (closed)
    ``LineOfInterest``, and may be treated accordingly.

    The class also supports holes - subregions properly contained inside the
    region that are not part of the region itself. These can be specified by
    the ``holes`` argument, and define the interior boundaries of the region.
    These interior boundaries are accessible via the ``.interior_boundaries``
    property, and the polygonal regions that make up the holes are accessible
    via the ``holes`` property.
    """

    def __init__(
        self,
        exterior_boundary: PointLikeList,
        holes: Sequence[PointLikeList] | None = None,
        name: str | None = None,
    ) -> None:
        """Create a new region of interest (RoI).

        Parameters
        ----------
        exterior_boundary : tuple of (x, y) pairs
            The points (in sequence) that make up the boundary of the region.
            At least three points must be provided.
        holes : sequence of sequences of (x, y) pairs, default None
            A sequence of items, where each item will be interpreted as the
            ``exterior_boundary`` of an internal hole within the region. See
            the ``holes`` argument to ``shapely.Polygon`` for details.
        name : str, optional
            Name of the RoI that is to be created. A default name will be
            inherited from the base class if not provided.

        See Also
        --------
        movement.roi.base.BaseRegionOfInterest : The base class that
            constructor arguments are passed to, and defaults are inherited
            from.

        """
        super().__init__(
            points=exterior_boundary, dimensions=2, holes=holes, name=name
        )

    @property
    def _default_plot_args(self) -> dict[str, Any]:
        return {
            **super()._default_plot_args,
            "facecolor": "lightblue",
            "edgecolor": "black",
        }

    @property
    def exterior_boundary(self) -> LineOfInterest:
        """The exterior boundary of this RoI."""
        return LineOfInterest(
            self.region.exterior.coords,
            loop=True,
            name=f"Exterior boundary of {self.name}",
        )

    @property
    def holes(self) -> tuple[PolygonOfInterest, ...]:
        """The interior holes of this RoI.

        Holes are regions properly contained within the exterior boundary of
        the RoI that are not part of the RoI itself (like the centre of a
        doughnut, for example). A region with no holes returns the empty tuple.
        """
        return tuple(
            PolygonOfInterest(
                int_boundary.coords, name=f"Hole {i} of {self.name}"
            )
            for i, int_boundary in enumerate(self.region.interiors)
        )

    @property
    def interior_boundaries(self) -> tuple[LineOfInterest, ...]:
        """The interior boundaries of this RoI.

        Interior boundaries are the boundaries of holes contained within the
        polygon.  A region with no holes returns the empty tuple.
        """
        return tuple(
            LineOfInterest(
                int_boundary.coords,
                loop=True,
                name=f"Interior boundary {i} of {self.name}",
            )
            for i, int_boundary in enumerate(self.region.interiors)
        )

    def _plot(
        self, fig: plt.Figure, ax: plt.Axes, **matplotlib_kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """Polygonal regions need to use patch to be plotted.

        In addition, ``matplotlib`` requires hole coordinates to be listed in
        the reverse orientation to the exterior boundary. Running
        ``numpy.flip`` on the exterior coordinates is a cheap way to ensure
        that we adhere to this convention, since our geometry is normalised
        upon creation, so this amounts to reversing the order of the
        coordinates.
        """
        exterior_boundary_as_path = PltPath(
            np.flip(np.asarray(self.exterior_boundary.coords), axis=0)
        )
        interior_boundaries_as_paths = [
            PltPath(np.asarray(ib.coords)) for ib in self.interior_boundaries
        ]
        path = PltPath.make_compound_path(
            exterior_boundary_as_path,
            *interior_boundaries_as_paths,
        )

        polygon_shape = PltPatch(path, **matplotlib_kwargs)
        ax.add_patch(polygon_shape)
        ax.autoscale_view(tight=True)
        return fig, ax
