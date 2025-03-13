"""Dataclasses containing layer styles for napari."""

from dataclasses import dataclass, field
from typing import Optional  # Added for type hinting

import numpy as np
import pandas as pd
from napari.utils.colormaps import ensure_colormap

DEFAULT_COLORMAP = "turbo"


@dataclass
class LayerStyle:
    """Base class for napari layer styles."""

    name: str  # Layer name
    properties: pd.DataFrame  # Dataframe containing layer properties
    visible: bool = True  # Visibility toggle
    blending: str = "translucent"  # Blending mode

    def as_kwargs(self) -> dict:
        """Return the style properties as a dictionary of kwargs."""
        return self.__dict__


@dataclass
class PointsStyle(LayerStyle):
    """Style properties for a napari Points layer."""

    symbol: str = "disc"  # Shape of points
    size: int = 10  # Size of the points
    border_width: int = 0  # Width of the border
    face_color: Optional[str] = None  # Color of the face
    face_color_cycle: Optional[list[tuple]] = None  # Cycle of colors
    face_colormap: str = DEFAULT_COLORMAP  # Default colormap
    text: dict = field(default_factory=lambda: {"visible": False})  # Text settings

    def set_color_by(self, prop: str, cmap: Optional[str] = None) -> None:
        """Set the face_color to a column in the properties DataFrame.

        Parameters
        ----------
        prop : str
            The column name in the properties DataFrame to color by.
        cmap : str, optional
            The name of the colormap to use, otherwise use the face_colormap.
        """
        if cmap is None:
            cmap = self.face_colormap

        self.face_color = prop  # Set face_color to the property column
        self.text["string"] = prop  # Update text settings
        n_colors = len(self.properties[prop].unique())  # Get unique values count
        self.face_color_cycle = _sample_colormap(n_colors, cmap)  # Generate colors


def _sample_colormap(n: int, cmap_name: str) -> list[tuple]:
    """Sample n equally-spaced colors from a napari colormap.

    This includes the endpoints of the colormap.
    
    Parameters
    ----------
    n : int
        Number of colors to sample.
    cmap_name : str
        Name of the colormap.

    Returns
    -------
    list[tuple]
        A list of sampled colors as tuples.
    """
    cmap = ensure_
