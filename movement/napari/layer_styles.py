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

    def set_color_by(
        self,
        property: str,
        properties_df: pd.DataFrame,
        cmap: str | None = None,
    ) -> None:
        """Color markers and text by a column in the properties DataFrame.

        Parameters
        ----------
        property : str
            The column name in the properties DataFrame to color by.
        properties_df : pd.DataFrame
            The properties DataFrame containing the data to color by.
            It should contain the column specified in `property`.
        cmap : str, optional
            The name of the colormap to use, otherwise use the face_colormap.

        """
        # Set points and text to be colored by selected property
        self.face_color = property
        if "color" in self.text:
            self.text["color"].update({"feature": property})
        else:
            self.text["color"] = {"feature": property}

        # Get color cycle
        if cmap is None:
            cmap = self.face_colormap
        n_colors = len(properties_df[property].unique())
        color_cycle = _sample_colormap(n_colors, cmap)

        # Set color cycle for points and text
        self.face_color_cycle = color_cycle
        self.text["color"].update({"colormap": color_cycle})

    def set_text_by(self, property: str) -> None:
        """Set the text property for the points layer.

        Parameters
        ----------
        property : str
            The column name in the properties DataFrame to use for text.

        """
        self.text["string"] = property


@dataclass
class TracksStyle(LayerStyle):
    """Style properties for a napari Tracks layer."""

    blending: str = "opaque"
    colormap: str = DEFAULT_COLORMAP
    color_by: str | None = "track_id"
    head_length: int = 0  # frames
    tail_length: int = 30  # frames
    tail_width: int = 2

    def set_color_by(self, property: str, cmap: str | None = None) -> None:
        """Color tracks by a column in the properties DataFrame.

        Parameters
        ----------
        property : str
            The column name in the properties DataFrame to color by.

        cmap : str, optional
            The name of the colormap to use. If not specified,
            DEFAULT_COLORMAP is used.

        """
        self.color_by = property

        # Overwrite colormap if specified
        if cmap is not None:
            self.colormap = cmap


@dataclass
class BoxesStyle(LayerStyle):
    """Style properties for a napari Shapes layer."""

    edge_width: int = 3
    opacity: float = 1.0
    shape_type: str = "rectangle"
    face_color: str = "#FFFFFF00"  # transparent face
    edge_colormap: str = DEFAULT_COLORMAP
    text: dict = field(
        default_factory=lambda: {
            "visible": True,  # default visible text for bboxes
            "anchor": "lower_left",
            "translation": 5,  # pixels
        }
    )

    def set_color_by(
        self,
        property: str,
        properties_df: pd.DataFrame,
        cmap: str | None = None,
    ) -> None:
        """Color boxes and text by chosen column in the properties DataFrame.

        Parameters
        ----------
        property : str
            The column name in the properties DataFrame to color shape edges
            and associated text by.
        properties_df : pd.DataFrame
            The properties DataFrame containing the data for generating the
            colormap.
        cmap : str, optional
            The name of the colormap to use, otherwise use the edge_colormap.

        Notes
        -----
        The input property is expected to be a column in the properties
        dataframe and it is used to define the color of the text. A factorized
        version of the property ("<property>_factorized") is used to define the
        edges color, and is also expected to be present in the properties
        dataframe.

        """
        # Compute color cycle based on property
        if cmap is None:
            cmap = self.edge_colormap
        n_colors = len(properties_df[property].unique())
        color_cycle = _sample_colormap(n_colors, cmap)

        # Set color for edges and text
        self.edge_color = property + "_factorized"
        self.edge_color_cycle = color_cycle
        self.text["color"] = {"feature": property}
        self.text["color"].update({"colormap": color_cycle})

    def set_text_by(self, property: str) -> None:
        """Set the text property for the boxes layer.

        Parameters
        ----------
        property : str
            The column name in the properties DataFrame to use for text.

        """
        self.text["string"] = property


def _sample_colormap(n: int, cmap_name: str) -> list[tuple]:
    """Sample n equally-spaced colors from a napari colormap.

    This includes the endpoints of the colormap.
    """
    cmap = ensure_colormap(cmap_name)
    samples = np.linspace(0, len(cmap.colors) - 1, n).astype(int)
    return [tuple(cmap.colors[i]) for i in samples]
