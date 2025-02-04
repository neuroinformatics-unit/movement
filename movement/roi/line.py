"""1-dimensional lines of interest."""

from movement.roi.base import BaseRegionOfInterest, PointLikeList


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
