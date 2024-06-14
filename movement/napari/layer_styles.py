"""Dataclasses containing layer styles for napari."""

from dataclasses import dataclass, field
from typing import Optional

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

    name: str
    properties: pd.DataFrame
    visible: bool = True
    blending: str = "translucent"
    symbol: str = "disc"
    size: int = 10
    edge_width: int = 0
    face_color: Optional[str] = None
    face_color_cycle: Optional[list[tuple]] = None
    face_colormap: str = DEFAULT_COLORMAP
    text: dict = field(default_factory=lambda: {"visible": False})

    @staticmethod
    def _sample_colormap(n: int, cmap_name: str) -> list[tuple]:
        """Sample n equally-spaced colors from a napari colormap,
        including the endpoints.
        """
        cmap = ensure_colormap(cmap_name)
        samples = np.linspace(0, len(cmap.colors) - 1, n).astype(int)
        return [tuple(cmap.colors[i]) for i in samples]

    def set_color_by(self, prop: str, cmap: str) -> None:
        """Set the face_color to a column in the properties DataFrame."""
        self.face_color = prop
        self.text["string"] = prop
        n_colors = len(self.properties[prop].unique())
        self.face_color_cycle = self._sample_colormap(n_colors, cmap)


@dataclass
class TracksStyle(LayerStyle):
    """Style properties for a napari Tracks layer."""

    name: str
    properties: pd.DataFrame
    tail_width: int = 5
    tail_length: int = 60
    head_length: int = 0
    color_by: str = "track_id"
    colormap: str = DEFAULT_COLORMAP
    visible: bool = True
    blending: str = "translucent"

    def set_color_by(self, prop: str, cmap: str) -> None:
        """Set the color_by to a column in the properties DataFrame."""
        self.color_by = prop
        self.colormap = cmap
