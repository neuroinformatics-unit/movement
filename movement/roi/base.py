"""Class for representing 1- or 2-dimensional regions of interest (RoIs)."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
from numpy.typing import ArrayLike
from shapely.coords import CoordinateSequence

from movement.utils.broadcasting import broadcastable_method
from movement.utils.logging import logger
from movement.utils.vector import compute_signed_angle_2d

LineLike: TypeAlias = shapely.LinearRing | shapely.LineString
PointLike: TypeAlias = list[float] | tuple[float, ...]
PointLikeList: TypeAlias = Sequence[PointLike] | np.ndarray
RegionLike: TypeAlias = shapely.Polygon
SupportedGeometry: TypeAlias = LineLike | RegionLike


class BaseRegionOfInterest:
    """Base class for regions of interest (RoIs).

    Regions of interest can be either 1 or 2 dimensional, and are represented
    by corresponding ``shapely.Geometry`` objects.

    To avoid the complexities of subclassing ``shapely`` objects (due to them
    relying on ``__new__()`` methods, see
    https://github.com/shapely/shapely/issues/1233), we simply wrap the
    relevant ``shapely`` object in the ``_shapely_geometry`` attribute of the
    class. This is accessible via the property ``region``. This also allows us
    to forbid certain operations and make the manipulation of ``shapely``
    objects more user friendly.

    Although this class can be instantiated directly, it is not designed for
    this. Its primary purpose is to reduce code duplication.

    Notes
    -----
    A region of interest includes the points that make up its boundary and the
    points contained in its interior. This convention means that points inside
    the region will be treated as having zero distance to the region, and the
    approach vector from these points to the region will be the null vector.

    This may be undesirable in certain situations, when we explicitly want to
    find the distance of a point to the boundary of a region, for example. To
    accommodate this, most methods of this class accept a keyword argument that
    forces the method to perform all computations using only the boundary of
    the region, rather than the region itself. For polygons, this will force
    the method in question to only consider distances or closest points on the
    segments that form the (interior and exterior) boundaries. For segments,
    the boundary is considered to be just the two endpoints of the segment.

    """

    __default_name: str = "Un-named region"

    _name: str | None
    _shapely_geometry: SupportedGeometry

    @property
    def _default_plot_args(self) -> dict[str, Any]:
        """Define default plotting arguments used when drawing the region.

        This argument is used inside ``self.plot``, which is implemented in the
        base class.

        This is implemented as a property for two reasons;
        - To ensure that the defaults can be set in a single place within the
        class definition,
        - To allow for easy overwriting in subclasses, which will be necessary
        given lines and polygons must be plotted differently,
        - In future, allows us to customise the defaults on a per-region basis
        (e.g., default labels can inherit ``self.name``).
        """
        kwargs = {}
        if self.name:
            kwargs["label"] = self.name

        return kwargs

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

    @staticmethod
    def _boundary_angle_computation(
        position: xr.DataArray,
        reference_vector: xr.DataArray | np.ndarray,
        how_to_compute_vector_to_region: Callable[
            [xr.DataArray], xr.DataArray
        ],
        in_degrees: bool = False,
    ) -> xr.DataArray:
        """Perform a boundary angle computation.

        Intended for internal use when conducting boundary angle computations,
        to reduce code duplication. All boundary angle computations involve two
        parts:

        - From some given spatial position data, compute the "vector towards
        the region". This is typically the approach vector, but might also be
        the normal vector if we are dealing with a segment or the plane
        supported by a segment.
        - Compute the signed angle between the "vector towards the region" and
        some given reference vector, which may be constant or varying in time
        (such as an animal's heading or forward vector).

        As such, we generalise the process into this internal method, and
        provide more explicit wrappers to the user to make working with the
        methods easier.

        Parameters
        ----------
        position : xarray.DataArray
            Spatial position data, that is passed to
            ``how_to_compute_vector_to_region`` and used to compute the
            "vector to the region".
        reference_vector : xarray.DataArray | np.ndarray
            Constant or time-varying vector to take signed angle with the
            "vector to the region".
        how_to_compute_vector_to_region : Callable
            How to compute the "vector to the region" from ``position``.
        in_degrees : bool
            If ``True``, angles are returned in degrees. Otherwise angles are
            returned in radians. Default ``False``.

        """
        vec_to_segment = how_to_compute_vector_to_region(position)

        angles = compute_signed_angle_2d(vec_to_segment, reference_vector)
        if in_degrees:
            angles = np.rad2deg(angles)
        return angles

    @staticmethod
    def _reassign_space_dim(
        da: xr.DataArray,
        old_dimension: Hashable,
    ) -> xr.DataArray:
        """Rename a computed dimension 'space' and assign coordinates.

        Intended for internal use when chaining ``DataArray``-broadcastable
        operations together. In some instances, the outputs drop the spatial
        coordinates, or the "space" axis is returned under a different name.
        This needs to be corrected before further computations can be
        performed.

        Parameters
        ----------
        da : xarray.DataArray
            ``DataArray`` lacking a "space" dimension, that is to be assigned.
        old_dimension : Hashable
            The dimension that should be renamed to "space", and reassigned
            coordinates.

        """
        return da.rename({old_dimension: "space"}).assign_coords(
            {
                "space": ["x", "y"]
                if len(da[old_dimension]) == 2
                else ["x", "y", "z"]
            }
        )

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
            raise logger.error(
                ValueError(
                    f"Need at least {dimensions + 1} points to define a "
                    f"{dimensions}D region (got {len(points)})."
                )
            )
        elif dimensions < 1 or dimensions > 2:
            raise logger.error(
                ValueError(
                    "Only regions of interest of dimension 1 or 2 "
                    f"are supported (requested {dimensions})"
                )
            )
        elif dimensions == 1 and len(points) < 3 and closed:
            raise logger.error(
                ValueError("Cannot create a loop from a single line segment.")
            )
        if dimensions == 2:
            self._shapely_geometry = shapely.Polygon(shell=points, holes=holes)
        else:
            self._shapely_geometry = (
                shapely.LinearRing(coordinates=points)
                if closed
                else shapely.LineString(coordinates=points)
            )
            self._shapely_geometry = shapely.normalize(self._shapely_geometry)

    def __repr__(self) -> str:  # noqa: D105
        return str(self)

    def __str__(self) -> str:  # noqa: D105
        display_type = "-gon" if self.dimensions > 1 else " line segment(s)"
        n_points = len(self.coords) - 1
        return (
            f"{self.__class__.__name__} {self.name} "
            f"({n_points}{display_type})\n"
        ) + " -> ".join(f"({c[0]}, {c[1]})" for c in self.coords)

    def _plot(
        self, fig: plt.Figure, ax: plt.Axes, **matplotlib_kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        raise NotImplementedError("_plot must be implemented by subclass.")

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
        self, point: ArrayLike, boundary_only: bool = False
    ) -> float:
        """Compute the distance from the region to a point.

        Parameters
        ----------
        point : ArrayLike
            Coordinates of a point, from which to find the nearest point in the
            region defined by ``self``.
        boundary_only : bool, optional
            If ``True``, compute the distance from ``point`` to the boundary of
            the region, rather than the closest point belonging to the region.
            Default ``False``.

        Returns
        -------
        float
            Euclidean distance from the ``point`` to the (closest point on the)
            region.

        """
        region_to_consider = (
            self.region.boundary if boundary_only else self.region
        )
        return shapely.distance(region_to_consider, shapely.Point(point))

    @broadcastable_method(
        only_broadcastable_along="space", new_dimension_name="nearest point"
    )
    def compute_nearest_point_to(
        self, /, position: ArrayLike, boundary_only: bool = False
    ) -> np.ndarray:
        """Compute (one of) the nearest point(s) in the region to ``position``.

        If there are multiple equidistant points, one of them is returned.

        Parameters
        ----------
        position : ArrayLike
            Coordinates of a point, from which to find the nearest point in the
            region.
        boundary_only : bool, optional
            If ``True``, compute the nearest point to ``position`` that is on
            the  boundary of ``self``. Default ``False``.

        Returns
        -------
        np.ndarray
            Coordinates of the point on ``self`` that is closest to
            ``position``.

        """
        region_to_consider = (
            self.region.boundary if boundary_only else self.region
        )
        # shortest_line returns a line from 1st arg to 2nd arg,
        # therefore the point on self is the 0th coordinate
        return np.array(
            shapely.shortest_line(
                region_to_consider, shapely.Point(position)
            ).coords[0]
        )

    @broadcastable_method(
        only_broadcastable_along="space", new_dimension_name="vector to"
    )
    def compute_approach_vector(
        self,
        point: ArrayLike,
        boundary_only: bool = False,
        unit: bool = False,
    ) -> np.ndarray:
        """Compute the approach vector from a ``point`` to the region.

        The approach vector is defined as the vector directed from the
        ``point`` provided, to the closest point that belongs to the region.

        Parameters
        ----------
        point : ArrayLike
            Coordinates of a point to compute the vector to (or from) the
            region.
        boundary_only : bool
            If ``True``, the approach vector to the boundary of the region is
            computed. Default ``False``.
        unit : bool
            If ``True``, the approach vector is returned normalised, otherwise
            it is not normalised. Default is ``False``.

        Returns
        -------
        np.ndarray
            Approach vector from the point to the region.

        See Also
        --------
        compute_allocentric_angle_to_nearest_point :
            Relies on this method to compute the approach vector.
        compute_egocentric_angle_to_nearest_point :
            Relies on this method to compute the approach vector.

        """
        region_to_consider = (
            self.region.boundary if boundary_only else self.region
        )

        # "point to region" by virtue of order of arguments to shapely call
        directed_line = shapely.shortest_line(
            shapely.Point(point), region_to_consider
        )

        approach_vector = np.array(directed_line.coords[1]) - np.array(
            directed_line.coords[0]
        )
        if unit:
            norm = np.sqrt(np.sum(approach_vector**2))
            # Cannot normalise the 0 vector
            if not np.isclose(norm, 0.0):
                approach_vector /= norm
        return approach_vector

    def compute_allocentric_angle_to_nearest_point(
        self,
        position: xr.DataArray,
        boundary_only: bool = False,
        in_degrees: bool = False,
        reference_vector: np.ndarray | xr.DataArray = None,
    ) -> xr.DataArray:
        """Compute the allocentric angle to the nearest point in the region.

        With the term "allocentric", we indicate that we are measuring angles
        with respect to a reference frame that is fixed relative to the
        experimental/camera setup. By default, we assume this is the positive
        x-axis of the coordinate system in which ``position`` is.

        The allocentric angle is the :func:`signed angle\
        <movement.utils.vector.compute_signed_angle_2d>` between the approach
        vector and a given reference vector.

        Parameters
        ----------
        position : xarray.DataArray
            ``DataArray`` of spatial positions.
        boundary_only : bool
            If ``True``, the allocentric angle to the closest boundary point of
            the region is computed. Default ``False``.
        in_degrees : bool
            If ``True``, angles are returned in degrees. Otherwise angles are
            returned in radians. Default ``False``.
        reference_vector : ArrayLike | xr.DataArray
            The reference vector to be used. Dimensions must be compatible with
            the argument of the same name that is passed to
            :func:`compute_signed_angle_2d`. Default ``(1., 0.)``.

        See Also
        --------
        compute_approach_vector :
            The method used to compute the approach vector.
        compute_egocentric_angle_to_nearest_point :
            Related class method for computing the egocentric angle to the
            region.
        movement.utils.vector.compute_signed_angle_2d :
            The underlying function used to compute the signed angle between
            the approach vector and the reference vector.

        """
        if reference_vector is None:
            reference_vector = np.array([[1.0, 0.0]])

        return self._boundary_angle_computation(
            position=position,
            reference_vector=reference_vector,
            how_to_compute_vector_to_region=lambda p: self._reassign_space_dim(
                self.compute_approach_vector(
                    p, boundary_only=boundary_only, unit=False
                ),
                "vector to",
            ),
            in_degrees=in_degrees,
        )

    def compute_egocentric_angle_to_nearest_point(
        self,
        direction: xr.DataArray,
        position: xr.DataArray,
        boundary_only: bool = False,
        in_degrees: bool = False,
    ) -> xr.DataArray:
        """Compute the egocentric angle to the nearest point in the region.

        With the term "egocentric", we indicate that we are measuring angles
        with respect to a reference frame that is varying in time relative to
        the experimental/camera setup.

        The egocentric angle is the signed angle between the approach vector
        and a ``direction`` vector (examples include the forward vector of
        a given individual, or the velocity vector of a given point).

        Parameters
        ----------
        direction : xarray.DataArray
            An array of vectors representing a given direction,
            e.g., the forward vector(s).
        position : xarray.DataArray
            `DataArray` of spatial positions, considered the origin of the
            ``direction`` vector.
        boundary_only : bool
            Passed to ``compute_approach_vector`` (see Notes). Default
            ``False``.
        in_degrees : bool
            If ``True``, angles are returned in degrees. Otherwise angles are
            returned in radians. Default ``False``.

        See Also
        --------
        compute_allocentric_angle_to_nearest_point :
            Related class method for computing the egocentric angle to the
            region.
        compute_approach_vector :
            The method used to compute the approach vector.
        movement.utils.vector.compute_signed_angle_2d :
            The underlying function used to compute the signed angle between
            the approach vector and the reference vector.

        """
        return self._boundary_angle_computation(
            position=position,
            reference_vector=direction,
            how_to_compute_vector_to_region=lambda p: self._reassign_space_dim(
                self.compute_approach_vector(
                    p, boundary_only=boundary_only, unit=False
                ),
                "vector to",
            ),
            in_degrees=in_degrees,
        )

    def plot(
        self, ax: plt.Axes | None = None, **matplotlib_kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the region of interest on a new or existing axis.

        Parameters
        ----------
        ax : plt.Axes, optional
            ``matplotlib.pyplot.Axes`` object to draw the region on. A new
            ``Figure`` and ``Axes`` will be created if not provided.
        matplotlib_kwargs : Any
            Keyword arguments passed to the ``matplotlib.pyplot`` plotting
            function.

        """
        for arg, default in self._default_plot_args.items():
            if arg not in matplotlib_kwargs:
                matplotlib_kwargs[arg] = default

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        return self._plot(fig, ax, **matplotlib_kwargs)
