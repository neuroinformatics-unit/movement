"""Dataclasses containing layer styles for napari."""

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from napari.layers import Shapes
from napari.utils.color import ColorValue
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
    """Style properties for a napari Shapes layer containing bounding boxes."""

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


@dataclass
class RegionsStyle(LayerStyle):
    """Style properties for a napari Shapes layer containing regions.

    The same ``color`` is applied to faces, edges, and text.
    The face color opacity is hardcoded to 0.25, while edges and text
    colors are opaque.
    """

    name: str = "Regions"
    color: str | tuple = "red"
    edge_width: float = 5.0
    opacity: float = 1.0  # applies to the whole layer
    text: dict = field(
        default_factory=lambda: {
            "visible": True,
            "anchor": "center",
        }
    )

    @property
    def face_color(self) -> ColorValue:
        """Return the face color with transparency applied."""
        color = ColorValue(self.color)
        color[-1] = 0.25  # this is hardcoded for now
        return color

    @property
    def edge_and_text_color(self) -> ColorValue:
        """Return the opaque color for edges and text."""
        color = ColorValue(self.color)
        color[-1] = 1.0
        return color

    def color_current_shape(self, layer: Shapes) -> None:
        """Color the current shape in a napari Shapes layer.

        napari uses current_* for new shapes.
        """
        # Only proceed if there are valid selected shapes
        if hasattr(layer, "selected_data") and layer.selected_data:
            valid_selected = {
                i for i in layer.selected_data if 0 <= i <= len(layer.data) - 1
            }
            if not valid_selected:
                return

        layer.current_face_color = self.face_color
        layer.current_edge_color = self.edge_and_text_color
        layer.current_edge_width = self.edge_width

    def color_all_shapes(self, layer: Shapes) -> None:
        """Color all shapes in a napari Shapes layer, including new ones."""
        n_shapes = len(layer.data)
        if n_shapes > 0:
            layer.face_color = [self.face_color] * n_shapes
            layer.edge_color = [self.edge_and_text_color] * n_shapes
            layer.edge_width = [self.edge_width] * n_shapes

            # Set text properties
            text_dict = layer.text.dict()
            text_dict.update(self.text)
            layer.text = text_dict
            layer.text.color = self.edge_and_text_color
            layer.text.string = "{name}"

        self.color_current_shape(layer)


@dataclass
class RegionsColorManager:
    """Manages colors for Regions layers.

    It makes sure that Regions layers are each assigned a color
    deterministically based on the layer name (using a hash), sampled
    from a napari colormap. This ensures the same layer name always gets
    the same color, even after deletion and recreation.
    """

    cmap_name: str = "tab10"
    n_colors: int = 10
    _color_cache: dict = field(default_factory=dict)
    colors: list = field(init=False)

    def __post_init__(self):
        """Initialize the colors after the dataclass is created."""
        self.colors = _sample_colormap(self.n_colors, self.cmap_name)

    # Pattern for default region layer names: "Regions" or "Regions [N]"
    _region_pattern = re.compile(r"^Regions(?: \[(\d+)\])?$")

    def get_color_for_layer(self, layer_name: str) -> tuple:
        """Get a deterministic color for a layer based on its name.

        For default region layer names ("Regions", "Regions [1]", etc.),
        colors are assigned sequentially to avoid collisions.
        For custom names, uses MD5 hash for deterministic color selection
        across Python sessions. Results are cached for efficiency.

        Parameters
        ----------
        layer_name : str
            The name of the layer.

        Returns
        -------
        tuple
            The RGBA color tuple for the layer.

        """
        if layer_name not in self._color_cache:
            color_index = self._get_color_index(layer_name)
            self._color_cache[layer_name] = self.colors[color_index]

        return self._color_cache[layer_name]

    def _get_color_index(self, layer_name: str) -> int:
        """Get the color index for a layer name.

        Sequential indices for default names, hash-based for custom names.
        """
        match = self._region_pattern.match(layer_name)
        if match:
            # "Regions" → 0, "Regions [1]" → 1, "Regions [2]" → 2, etc.
            suffix = match.group(1)
            index = int(suffix) if suffix else 0
            return index % len(self.colors)

        # Fall back to MD5 hash for custom names
        hash_digest = hashlib.md5(layer_name.encode()).hexdigest()
        return int(hash_digest, 16) % len(self.colors)


def _sample_colormap(n: int, cmap_name: str) -> list[tuple]:
    """Sample n equally-spaced colors from a napari colormap.

    This includes the endpoints of the colormap.
    """
    cmap = ensure_colormap(cmap_name)
    samples = np.linspace(0, len(cmap.colors) - 1, n).astype(int)
    return [tuple(cmap.colors[i]) for i in samples]
