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
    text: dict = field(
        default_factory=lambda: {
            "visible": False,
            "anchor": "lower_left",
            # it actually displays the text in the lower
            # _right_ corner of the marker
            "translation": 5,  # pixels
        }
    )

    def set_color_by(self, prop: str, cmap: str | None = None) -> None:
        """Color markers and text by a column in the properties DataFrame.

        Parameters
        ----------
        prop : str
            The column name in the properties DataFrame to color by.
        cmap : str, optional
            The name of the colormap to use, otherwise use the face_colormap.

        """
        # Set points and text to be colored by selected property
        self.face_color = prop
        if "color" in self.text:
            self.text["color"].update({"feature": prop})
        else:
            self.text["color"] = {"feature": prop}

        # Get color cycle
        if cmap is None:
            cmap = self.face_colormap
        n_colors = len(self.properties[prop].unique())
        color_cycle = _sample_colormap(n_colors, cmap)

        # Set color cycle for points and text
        self.face_color_cycle = color_cycle
        self.text["color"].update({"colormap": color_cycle})

    def set_text_by(self, prop: str) -> None:
        """Set the text property for the points layer.

        Parameters
        ----------
        prop : str
            The column name in the properties DataFrame to use for text.

        """
        self.text["string"] = prop


def _sample_colormap(n: int, cmap_name: str) -> list[tuple]:
    """Sample n equally-spaced colors from a napari colormap.

    This includes the endpoints of the colormap.
    """
    cmap = ensure_colormap(cmap_name)
    samples = np.linspace(0, len(cmap.colors) - 1, n).astype(int)
    return [tuple(cmap.colors[i]) for i in samples]
