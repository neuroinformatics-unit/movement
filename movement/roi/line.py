"""1-dimensional lines of interest."""

from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from movement.kinematics import compute_forward_vector_angle
from movement.roi.base import (
    ApproachVectorDirections,
    AwayFromRegion,
    BaseRegionOfInterest,
    PointLikeList,
    TowardsRegion,
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

    @broadcastable_method(
        only_broadcastable_along="space", new_dimension_name="normal"
    )
    def normal(self, on_same_side_as: ArrayLike = (0.0, 0.0)) -> np.ndarray:
        """Compute the unit normal to this line.

        There are always two normal vectors to chose from. The normal vector
        that points to the same side of the segment as that which the point
        ``on_same_side_as`` lies on, is the returned normal vector.

        Parameters
        ----------
        on_same_side_as : ArrayLike
            Point in the (x,y) plane to orientate the returned normal vector
            towards.

        """
        on_same_side_as = np.array(on_same_side_as)

        parallel = np.array(self.region.coords[1]) - np.array(
            self.region.coords[0]
        )
        normal = np.array([parallel[1], -parallel[0]])
        normal /= np.sqrt(np.sum(normal**2))

        if np.sum(on_same_side_as * (normal - self.region.coords[0])) < 0:
            normal *= -1.0
        return normal

    def compute_angle_to_support_plane_of_segment(
        self,
        data: xr.DataArray,
        left_keypoint: Hashable,
        right_keypoint: Hashable,
        angle_rotates: Literal[
            "forward to normal", "normal to forward"
        ] = "normal to forward",
        camera_view: Literal["top_down", "bottom_up"] = "top_down",
        in_radians: bool = False,
        normal_direction: ApproachVectorDirections = TowardsRegion,
        position_keypoint: Hashable | Sequence[Hashable] | None = None,
    ) -> xr.DataArray:
        """Compute the signed angle between the normal and a forward vector.

        This method is identical to ``compute_egocentric_angle``, except that
        rather than the angle between the approach vector and a forward vector,
        the angle between the normal to the segment and the approach vector is
        returned.

        For finite segments, the normal to the infinite extension of the
        segment is used in the calculation.

        Parameters
        ----------
        data : xarray.DataArray
            `DataArray` of positions that has at least 3 dimensions; "time",
            "space", and ``keypoints_dimension``.
        left_keypoint : Hashable
            The left keypoint defining the forward vector, as passed to
            func:``compute_forward_vector_angle``.
        right_keypoint : Hashable
            The right keypoint defining the forward vector, as passed to
            func:``compute_forward_vector_angle``.
        angle_rotates : Literal["approach to forward", "forward to approach"]
            Direction of the signed angle returned. Default is
            ``"approach to forward"``.
        camera_view : Literal["top_down", "bottom_up"]
            Passed to func:`compute_forward_vector_angle`. Default
            ``"top_down"``.
        in_radians : bool
            If ``True``, angles are returned in radians. Otherwise angles are
            returned in degrees. Default ``False``.
        normal_direction : ApproachVectorDirections
            Direction to use for the normal vector. Default is
            ``"point to region"``.
        position_keypoint : Hashable | Sequence[Hashable], optional
            The keypoint defining the origin of the approach vector. If
            provided as a sequence, the average of all provided keypoints is
            used. By default, the centroid of ``left_keypoint`` and
            ``right_keypoint`` is used.

        """
        # Default to centre of left and right keypoints for position,
        # if not provided.
        if position_keypoint is None:
            position_keypoint = [left_keypoint, right_keypoint]

        normal = self._vector_from_centroid_of_keypoints(
            data,
            position_keypoint=position_keypoint,
            renamed_dimension="normal",
            which_method="normal",
        )
        # The normal from the point to the region is the same as the normal
        # that points into the opposite side of the segment.
        if normal_direction == TowardsRegion:
            normal *= -1.0
        elif normal_direction != AwayFromRegion:
            raise ValueError(
                f"Unknown convention for normal vector: {normal_direction}"
            )

        # Translate the more explicit convention used here into the convention
        # used by our backend functions.
        rotation_angle: Literal["ref to forward", "forward to ref"] = (
            angle_rotates.replace("normal", "ref")  # type: ignore
        )
        if rotation_angle not in ["ref to forward", "forward to ref"]:
            raise ValueError(f"Unknown angle convention: {angle_rotates}")

        return compute_forward_vector_angle(
            data,
            left_keypoint=left_keypoint,
            right_keypoint=right_keypoint,
            reference_vector=normal,
            camera_view=camera_view,
            in_radians=in_radians,
            angle_rotates=rotation_angle,
        )
