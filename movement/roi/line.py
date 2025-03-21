"""1-dimensional lines of interest."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.roi.base import (
    BaseRegionOfInterest,
    PointLikeList,
)
from movement.utils.broadcasting import broadcastable_method


class LineOfInterest(BaseRegionOfInterest):
    """Representation of boundaries or other lines of interest.

    This class can be used to represent boundaries or other internal divisions
    of the area in which the experimental data was gathered. These might
    include segments of a wall that are removed partway through a behavioural
    study, or coloured marking on the floor of the experimental enclosure that
    have some significance. Instances of this class also constitute the
    boundary of two-dimensional regions (polygons) of interest.

    An instance of this class can be used to represent these "one dimensional
    regions" (lines of interest, LoIs) in an analysis. The basic usage is to
    construct an instance of this class by passing in a list of points, which
    will then be joined (in sequence) by straight lines between consecutive
    pairs of points, to form the LoI that is to be studied.
    """

    def __init__(
        self,
        points: PointLikeList,
        loop: bool = False,
        name: str | None = None,
    ) -> None:
        """Create a new line of interest (LoI).

        Parameters
        ----------
        points : tuple of (x, y) pairs
            The points (in sequence) that make up the line segment. At least
            two points must be provided.
        loop : bool, default False
            If True, the final point in ``points`` will be connected by an
            additional line segment to the first, creating a closed loop.
            (See Notes).
        name : str, optional
            Name of the LoI that is to be created. A default name will be
            inherited from the base class if not provided, and
            defaults are inherited from.

        Notes
        -----
        The constructor supports 'rings' or 'closed loops' via the ``loop``
        argument. However, if you want to define an enclosed region for your
        analysis, we recommend you create a ``PolygonOfInterest`` and use
        its ``boundary`` property instead.

        See Also
        --------
        movement.roi.base.BaseRegionOfInterest
            The base class that constructor arguments are passed to.

        """
        super().__init__(points, dimensions=1, closed=loop, name=name)

    def _plot(
        self, fig: plt.Figure, ax: plt.Axes, **matplotlib_kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """LinesOfInterest can simply be plotted as lines."""
        ax.plot(
            [c[0] for c in self.coords],
            [c[1] for c in self.coords],
            **matplotlib_kwargs,
        )
        return fig, ax

    @broadcastable_method(
        only_broadcastable_along="space", new_dimension_name="normal"
    )
    def normal(self, on_same_side_as: ArrayLike = (0.0, 0.0)) -> np.ndarray:
        """Compute the unit normal to this line.

        The unit normal is a vector perpendicular to the input line
        whose norm is equal to 1. The direction of the normal vector
        is not fully defined: the line divides the 2D plane in two
        halves, and the normal could be pointing to either of the half-planes.
        For example, a horizontal line divides the 2D plane in a
        bottom and a top half-plane, and we can choose whether
        the normal points "upwards" or "downwards". We use a sample
        point to define the half-plane the normal vector points to.

        If this is a multi-segment line, the method raises an error.

        Parameters
        ----------
        on_same_side_as : ArrayLike
            A sample point in the (x,y) plane the normal is in. If multiple
            points are given, one normal vector is returned for each point
            given. By default, the origin is used.

        Raises
        ------
        ValueError : When the normal is requested for a multi-segment geometry.

        """
        # A multi-segment geometry always has at least 3 coordinates.
        if len(self.coords) > 2:
            raise ValueError(
                "Normal is not defined for multi-segment geometries."
            )

        on_same_side_as = np.array(on_same_side_as)

        parallel_to_line = np.array(self.region.coords[1]) - np.array(
            self.region.coords[0]
        )
        normal = np.array([parallel_to_line[1], -parallel_to_line[0]])
        normal /= np.sqrt(np.sum(normal**2))

        if np.dot((on_same_side_as - self.region.coords[0]), normal) < 0:
            normal *= -1.0
        return normal

    def compute_angle_to_normal(
        self,
        direction: xr.DataArray,
        position: xr.DataArray,
        in_degrees: bool = False,
    ) -> xr.DataArray:
        """Compute the angle between the normal to the segment and a direction.

        The returned angle is the signed angle between the normal to the
        segment and the ``direction`` vector(s) provided.

        Parameters
        ----------
        direction : xarray.DataArray
            An array of vectors representing a given direction,
            e.g., the forward vector(s).
        position : xr.DataArray
            Spatial positions, considered the origin of the ``direction``.
        in_degrees : bool
            If ``True``, angles are returned in degrees. Otherwise angles are
            returned in radians. Default ``False``.

        See Also
        --------
        movement.utils.vector.compute_signed_angle_2d :
            For the definition of the signed angle between two vectors.

        """
        return self._boundary_angle_computation(
            position=position,
            reference_vector=direction,
            how_to_compute_vector_to_region=lambda p: self._reassign_space_dim(
                -1.0 * self.normal(p), "normal"
            ),
            in_degrees=in_degrees,
        )
