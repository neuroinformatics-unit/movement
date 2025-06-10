import numpy as np


class ROI:
    """Base class for regions of interest."""

    def __init__(self, coordinates):
        self.coordinates = np.asarray(coordinates)
        self.properties = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(coordinates.shape={self.coordinates.shape})"


class Line(ROI):
    """A line ROI defined by a series of points."""

    def __init__(self, coordinates):
        super().__init__(coordinates)
        if len(self.coordinates.shape) != 2:
            raise ValueError("Line coordinates must be 2D array of points")


class Polygon(ROI):
    """A polygon ROI defined by vertices."""

    def __init__(self, coordinates):
        super().__init__(coordinates)
        if len(self.coordinates.shape) != 2:
            raise ValueError(
                "Polygon coordinates must be 2D array of vertices"
            )
        # Ensure the polygon is closed
        if not np.array_equal(self.coordinates[0], self.coordinates[-1]):
            self.coordinates = np.vstack(
                [self.coordinates, self.coordinates[0]]
            )
