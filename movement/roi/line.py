"""1-dimensional lines of interest."""

from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.roi.base import (
    BaseRegionOfInterest,
    PointLikeList,
)
from movement.utils.broadcasting import broadcastable_method
from movement.utils.vector import compute_signed_angle_2d


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

    @broadcastable_method(
        only_broadcastable_along="space", new_dimension_name="normal"
    )
    def normal(self, on_same_side_as: ArrayLike = (0.0, 0.0)) -> np.ndarray:
        """Compute the unit normal to this line.

        The unit normal is a vector perpendicular to the input line
        whose norm is equal to 1. The direction of the normal vector
        is not fully defined: the line divides the 2D plane in two
        halves, and the normal could be pointing to either of the half-planes.
        For example, an horizontal line divides the 2D plane in a
        bottom and a top half-plane, and we can choose whether
        the normal points "upwards" or "downwards". We use a sample
        point to define the half-plane the normal vector points to.

        Parameters
        ----------
        on_same_side_as : ArrayLike
            A sample point in the (x,y) plane the normal is in. By default, the
            origin is used.

        """
        on_same_side_as = np.array(on_same_side_as)

        parallel_to_line = np.array(self.region.coords[1]) - np.array(
            self.region.coords[0]
        )
        normal = np.array([parallel_to_line[1], -parallel_to_line[0]])
        normal /= np.sqrt(np.sum(normal**2))

        if np.dot((on_same_side_as - self.region.coords[0]), normal) < 0:
            normal *= -1.0
        return normal

    def compute_angle_to_support_plane(
        self,
        forward_vector: xr.DataArray,
        position: xr.DataArray,
        angle_rotates: Literal[
            "forward to normal", "normal to forward"
        ] = "normal to forward",
        in_degrees: bool = False,
    ) -> xr.DataArray:
        """Compute the signed angle between the normal and the forward vector.

        This method is identical to ``compute_egocentric_angle``, except that
        rather than the angle between the approach vector and the forward
        vector, the angle between the normal directed toward the segment and
        the forward vector is returned.

        Parameters
        ----------
        forward_vector : xarray.DataArray
            Forward vectors to take angle with.
        position : xr.DataArray
            Spatial positions, considered the origin of the ``forward_vector``.
        angle_rotates : Literal["forward to normal", "normal to forward"]
            Sign convention of the angle returned. Default is
            ``"normal to forward"``.
        in_degrees : bool
            If ``True``, angles are returned in degrees. Otherwise angles are
            returned in radians. Default ``False``.

        """
        # Normal from position to segment is the reverse of what normal returns
        normal = self._reassign_space_dim(
            -1.0 * self.normal(position), "normal"
        )

        # Translate the more explicit convention used here into the convention
        # used by our backend functions.
        if angle_rotates == "normal to forward":
            forward_as_left_operand = False
        elif angle_rotates == "forward to normal":
            forward_as_left_operand = True
        else:
            raise ValueError(f"Unknown angle convention: {angle_rotates}")

        angles = compute_signed_angle_2d(
            normal, forward_vector, v_as_left_operand=forward_as_left_operand
        )
        if in_degrees:
            angles = np.rad2deg(angles)
        return angles
