"""Class for representing 1- or 2-dimensional regions of interest (RoIs)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import shapely
from numpy.typing import ArrayLike
from shapely.coords import CoordinateSequence

from movement.kinematics import compute_forward_vector_angle
from movement.utils.broadcasting import broadcastable_method
from movement.utils.logging import log_error
from movement.utils.vector import compute_signed_angle_2d

if TYPE_CHECKING:
    import xarray as xr

LineLike: TypeAlias = shapely.LinearRing | shapely.LineString
PointLike: TypeAlias = list[float] | tuple[float, ...]
PointLikeList: TypeAlias = Sequence[PointLike]
RegionLike: TypeAlias = shapely.Polygon
SupportedGeometry: TypeAlias = LineLike | RegionLike


class BaseRegionOfInterest:
    """Base class for representing regions of interest (RoIs).

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

    __default_name: str = "Un-named region"

    _name: str | None
    _shapely_geometry: SupportedGeometry

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
        holes : sequence of sequences of (x, y) pairs, default None
            A sequence of items, where each item will be interpreted like
            ``points``. These items will be used to construct internal holes
            within the region. See the ``holes`` argument to
            ``shapely.Polygon`` for details. Ignored if ``dimensions`` is 1.
        name : str, default None
            Human-readable name to assign to the given region, for
            user-friendliness. Default name given is 'Un-named region' if no
            explicit name is provided.

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
    def contains_point(
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

    @broadcastable_method(only_broadcastable_along="space")
    def compute_distance_to(
        self, point: ArrayLike, boundary: bool = False
    ) -> float:
        """Compute the distance from the region to a point.

        Parameters
        ----------
        point : ArrayLike
            Coordinates of a point, from which to find the nearest point in the
            region defined by ``self``.
        boundary : bool, optional
            If True, compute the distance from ``point`` to the boundary of
            the region. Otherwise, the distance returned may be 0 for interior
            points (see Notes).

        Returns
        -------
        float
            Euclidean distance from the ``point`` to the (closest point on the)
            region.

        Notes
        -----
        A point within the interior of a region is considered to be a distance
        0 from the region. This is desirable for 1-dimensional regions, but may
        not be desirable for 2D regions. As such, passing the ``boundary``
        argument as ``True`` makes the method compute the distance from the
        ``point`` to the boundary of the region, even if ``point`` is in the
        interior of the region.

        See Also
        --------
        shapely.distance : Underlying used to compute the nearest point.

        """
        from_where = self.region.boundary if boundary else self.region
        return shapely.distance(from_where, shapely.Point(point))

    @broadcastable_method(
        only_broadcastable_along="space", new_dimension_name="nearest point"
    )
    def compute_nearest_point_to(
        self, /, position: ArrayLike, boundary: bool = False
    ) -> np.ndarray:
        """Compute the nearest point in the region to the ``position``.

        position : ArrayLike
            Coordinates of a point, from which to find the nearest point in the
            region defined by ``self``.
        boundary : bool, optional
            If True, compute the nearest point to ``position`` that is on the
            boundary of ``self``. Otherwise, the nearest point returned may be
            inside ``self`` (see Notes).

        Returns
        -------
        np.ndarray
            Coordinates of the point on ``self`` that is closest to
            ``position``.

        Notes
        -----
        This function computes the nearest point to ``position`` in the region
        defined by ``self``. This means that, given a ``position`` inside the
        region, ``position`` itself will be returned. To find the nearest point
        to ``position`` on the boundary of a region, pass the ``boundary`
        argument as ``True`` to this method. Take care though - the boundary of
        a line is considered to be just its endpoints.

        See Also
        --------
        shapely.shortest_line : Underlying used to compute the nearest point.

        """
        from_where = self.region.boundary if boundary else self.region
        # shortest_line returns a line from 1st arg to 2nd arg,
        # therefore the point on self is the 0th coordinate
        return np.array(
            shapely.shortest_line(from_where, shapely.Point(position)).coords[
                0
            ]
        )

    @broadcastable_method(
        only_broadcastable_along="space", new_dimension_name="vector to"
    )
    def compute_approach_vector(
        self,
        point: ArrayLike,
        boundary: bool = False,
        direction: Literal[
            "point to region", "region to point"
        ] = "point to region",
        unit: bool = True,
    ) -> np.ndarray:
        """Compute the approach vector a ``point`` to the region.

        The approach vector is defined as the vector directed from the
        ``point`` provided, to the closest point that belongs to the region.
        If ``point`` is within the region, the zero vector is returned (see
        Notes).

        Parameters
        ----------
        point : ArrayLike
            Coordinates of a point to compute the vector to (or from) the
            region.
        boundary : bool
            If True, finds the vector to the nearest point on the boundary of
            the region, instead of the nearest point within the region.
            (See Notes). Default is False.
        direction : Literal["point to region", "region to point"]
            Which direction the returned vector should point in. Default is
            "point to region".
        unit : bool
            If ``True``, the unit vector in the appropriate direction is
            returned, otherwise the displacement vector is returned.
            Default is ``True``.

        Returns
        -------
        np.ndarray
            Vector directed between the point and the region.

        Notes
        -----
        If given a ``point`` in the interior of the region, the vector from
        this ``point`` to the region is treated as the zero vector. The
        ``boundary`` argument can be used to force the method to find the
        distance from the ``point`` to the nearest point on the boundary of the
        region, if so desired. Note that a ``point`` on the boundary still
        returns the zero vector.

        """
        from_where = self.region.boundary if boundary else self.region

        # "point to region" by virtue of order of arguments to shapely call
        directed_line = shapely.shortest_line(shapely.Point(point), from_where)

        displacement_vector = np.array(directed_line.coords[1]) - np.array(
            directed_line.coords[0]
        )
        if direction == "region to point":
            displacement_vector *= -1.0
        if unit:
            norm = np.sqrt(np.sum(displacement_vector**2))
            # Cannot normalise the 0 vector
            if norm != 0.0:
                displacement_vector /= norm
        return displacement_vector

    def compute_allocentric_angle(
        self,
        data: xr.DataArray,
        position_keypoint: Hashable | Sequence[Hashable],
        angle_rotates: Literal[
            "approach to ref", "ref to approach"
        ] = "approach to ref",
        approach_direction: Literal[
            "point to region", "region to point"
        ] = "point to region",
        boundary: bool = False,
        in_radians: bool = False,
        keypoints_dimension: Hashable = "keypoints",
        reference_vector: ArrayLike | xr.DataArray = (1.0, 0.0),
    ) -> float:
        """Compute the allocentric angle to the region.

        The allocentric angle is the :func:`signed angle\
        <movement.utils.vector.compute_signed_angle_2d>` between the approach
        vector (directed from a point to the region) and a given reference
        vector. `angle_rotates`` can be used to reverse the sign convention of
        the returned angle.

        The approach vector is the vector from ``position_keypoints`` to the
        closest point within the region (or the closest point on the boundary
        of the region if ``boundary`` is set to ``True``), as determined by
        :func:`compute_approach_vector`. ``approach_direction`` can be used to
        reverse the direction convention of the approach vector, if desired.

        Parameters
        ----------
        data : xarray.DataArray
            `DataArray` of positions that has at least 3 dimensions; "time",
            "space", and ``keypoints_dimension``.
        position_keypoint : Hashable | Sequence[Hashable]
            The keypoint defining the origin of the approach vector. If
            provided as a sequence, the average of all provided keypoints is
            used.
        angle_rotates : Literal["approach to ref", "ref to approach"]
            Direction of the signed angle returned. Default is
            ``"approach to ref"``.
        approach_direction : Literal["point to region", "region to point"]
            Direction to use for the vector of closest approach. Default
            ``"point to region"``.
        boundary : bool
            Passed to ``compute_approach_vector`` (see Notes). Default
            ``False``.
        in_radians : bool
            If ``True``, angles are returned in radians. Otherwise angles are
            returned in degrees. Default ``False``.
        keypoints_dimension : Hashable
            The dimension of ``data`` along which the ``position_keypoint`` is
            located. Default ``"keypoints"``.
        reference_vector : ArrayLike | xr.DataArray
            The reference vector to be used. Dimensions must be compatible with
            the argument of the same name that is passed to
            :func:`compute_signed_angle_2d`. Default ``(1., 0.)``.

        Notes
        -----
        For a point ``p`` within (the interior of a) region of interest, the
        approach vector to the region is the zero vector, since the closest
        point to ``p`` inside the region is ``p`` itself. This behaviour may be
        undesirable for 2D polygonal regions (though is usually desirable for
        1D line-like regions). Passing the ``boundary`` argument causes this
        method to calculate the approach vector to the closest point on the
        boundary of this region, so ``p`` will not be considered the "closest
        point" to itself (unless of course, it is a point on the boundary).

        See Also
        --------
        ``compute_signed_angle_2d`` : The underlying function used to compute
        the signed angle between the approach vector and the reference vector.
        ``compute_egocentric_angle`` : Related class method for computing the
        egocentric angle to the region.

        """
        # Translate the more explicit convention used here into the convention
        # used by our backend functions.
        if angle_rotates == "ref to approach":
            ref_as_left_opperand = True
        elif angle_rotates == "approach to ref":
            ref_as_left_opperand = False
        else:
            raise ValueError(
                f"Cannot interpret angle convention: {angle_rotates}"
            )

        # If we are given multiple position keypoints, we take the average of
        # them all.
        position_data = data.sel(
            {keypoints_dimension: position_keypoint}
        ).mean(dim=keypoints_dimension)
        # Determine the approach vector, for all time-points.
        vector_to_region = self.compute_approach_vector(
            position_data,
            boundary=boundary,
            direction=approach_direction,
            unit=True,
        )

        # Then, compute signed angles at all time-points
        angles = compute_signed_angle_2d(
            vector_to_region,
            reference_vector,
            v_as_left_operand=ref_as_left_opperand,
        )
        if not in_radians:
            angles *= 180.0 / np.pi
        return angles

    def compute_egocentric_angle(
        self,
        data: xr.DataArray,
        left_keypoint: Hashable,
        right_keypoint: Hashable,
        angle_rotates: Literal[
            "approach to forward", "forward to approach"
        ] = "approach to forward",
        approach_direction: Literal[
            "point to region", "region to point"
        ] = "point to region",
        boundary: bool = False,
        camera_view: Literal["top_down", "bottom_up"] = "top_down",
        in_radians: bool = False,
        keypoints_dimension: Hashable = "keypoints",
        position_keypoint: Hashable | Sequence[Hashable] | None = None,
    ) -> xr.DataArray:
        """Compute the egocentric angle to the region.

        The egocentric angle is the signed angle between the approach vector
        (directed from a point towards the region) a forward direction
        (typically of a given individual or keypoint). ``angle_rotates`` can
        be used to reverse the sign convention of the returned angle.

        The forward vector is determined by ``left_keypoint``,
        ``right_keypoint``, and ``camera_view`` as per :func:`forward vector\
        <movement.kinematics.compute_forward_vector>`.

        The approach vector is the vector from ``position_keypoints`` to the
        closest point within the region (or the closest point on the boundary
        of the region if ``boundary`` is set to ``True``), as determined by
        :func:`compute_approach_vector`. ``approach_direction`` can be used to
        reverse the direction convention of the approach vector, if desired.

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
        approach_direction : Literal["point to region", "region to point"]
            Direction to use for the vector of closest approach. Default
            ``"point to region"``.
        boundary : bool
            Passed to ``compute_approach_vector`` (see Notes). Default
            ``False``.
        camera_view : Literal["top_down", "bottom_up"]
            Passed to func:`compute_forward_vector_angle`. Default
            ``"top_down"``.
        in_radians : bool
            If ``True``, angles are returned in radians. Otherwise angles are
            returned in degrees. Default ``False``.
        keypoints_dimension : Hashable
            The dimension of ``data`` along which the (left, right, and
            position) keypoints are located. Default ``"keypoints"``.
        position_keypoint : Hashable | Sequence[Hashable], optional
            The keypoint defining the origin of the approach vector. If
            provided as a sequence, the average of all provided keypoints is
            used. By default, the centroid of ``left_keypoint`` and
            ``right_keypoint`` is used.

        Notes
        -----
        For a point ``p`` within (the interior of a) region of interest, the
        approach vector to the region is the zero vector, since the closest
        point to ``p`` inside the region is ``p`` itself. This behaviour may be
        undesirable for 2D polygonal regions (though is usually desirable for
        1D line-like regions). Passing the ``boundary`` argument causes this
        method to calculate the approach vector to the closest point on the
        boundary of this region, so ``p`` will not be considered the "closest
        point" to itself (unless of course, it is a point on the boundary).

        See Also
        --------
        ``compute_forward_vector_angle`` : The underlying function used
            to compute the signed angle between the forward vector and the
            approach vector.

        """
        # Default to centre of left and right keypoints for position,
        # if not provided.
        if position_keypoint is None:
            position_keypoint = (left_keypoint, right_keypoint)
        # Translate the more explicit convention used here into the convention
        # used by our backend functions.
        rotation_angle: Literal["ref to forward", "forward to ref"] = (
            angle_rotates.replace("approach", "ref")  # type: ignore
        )

        # If we are given multiple position keypoints, we take the average of
        # them all.
        position_data = data.sel(
            {keypoints_dimension: position_keypoint}
        ).mean(dim=keypoints_dimension)
        # Determine the approach vector, for all time-points.
        vector_to_region = self.compute_approach_vector(
            position_data,
            boundary=boundary,
            direction=approach_direction,
            unit=True,
        )

        # Then, compute signed angles at all time-points
        return compute_forward_vector_angle(
            data,
            left_keypoint=left_keypoint,
            right_keypoint=right_keypoint,
            reference_vector=vector_to_region,
            camera_view=camera_view,
            in_radians=in_radians,
            angle_rotates=rotation_angle,
        )
