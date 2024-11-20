"""Dataclasses containing layer styles for napari."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from napari.utils.colormaps import ensure_colormap

DEFAULT_COLORMAP = "turbo"


@dataclass
class LayerStyle:
    """Base class for napari layer styles."""

    name: str
    properties: pd.DataFrame
    visible: bool = True
    blending: str = "translucent"

    def as_kwargs(self) -> dict:
        """Return the style properties as a dictionary of kwargs."""
        return self.__dict__


@dataclass
class PointsStyle(LayerStyle):
    """Style properties for a napari Points layer."""

    symbol: str = "disc"
    size: int = 10
    border_width: int = 0
    face_color: str | None = None
    face_color_cycle: list[tuple] | None = None
    face_colormap: str = DEFAULT_COLORMAP
    text: dict = field(default_factory=lambda: {"visible": False})

    def set_color_by(self, prop: str, cmap: str | None = None) -> None:
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
        self.face_color = prop
        self.text["string"] = prop
        n_colors = len(self.properties[prop].unique())
        self.face_color_cycle = _sample_colormap(n_colors, cmap)


def _sample_colormap(n: int, cmap_name: str) -> list[tuple]:
    """Sample n equally-spaced colors from a napari colormap.

    This includes the endpoints of the colormap.
    """
    cmap = ensure_colormap(cmap_name)
    samples = np.linspace(0, len(cmap.colors) - 1, n).astype(int)
    return [tuple(cmap.colors[i]) for i in samples]
