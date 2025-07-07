"""Unit tests for the LayerStyle and PointsStyle classes."""

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from movement.napari.layer_styles import (
    DEFAULT_COLORMAP,
    BoxesStyle,
    LayerStyle,
    PointsStyle,
    RoisStyle,
    TracksStyle,
    _sample_colormap,
)


@pytest.fixture
def sample_properties():
    """Fixture that provides a sample "properties" DataFrame."""
    data = {"category": ["A", "B", "A", "C", "B"], "value": [1, 2, 3, 4, 5]}
    return pd.DataFrame(data)


@pytest.fixture
def sample_layer_style():
    """Fixture that provides a sample LayerStyle or subclass instance."""

    def _sample_layer_style(layer_class):
        return layer_class(name="Layer1")

    return _sample_layer_style


@pytest.fixture
def default_style_attributes():
    """Fixture that provides expected attributes for LayerStyle and subclasses.

    It holds the default values we expect after initialisation.
    """
    return {
        # Shared attributes for LayerStyle and all its subclasses
        LayerStyle: {
            "name": "Layer1",  # as given in sample_layer_style
            "visible": True,
            "blending": "translucent",
        },
        # Additional attributes for PointsStyle
        PointsStyle: {
            "symbol": "disc",
            "size": 10,
            "border_width": 0,
            "face_color": None,
            "face_color_cycle": None,
            "face_colormap": DEFAULT_COLORMAP,
            "text": {
                "visible": False,
                "anchor": "lower_left",
                "translation": 5,
            },
        },
        # Additional attributes for TracksStyle
        TracksStyle: {
            "blending": "opaque",
            "colormap": DEFAULT_COLORMAP,
            "color_by": "track_id",
            "head_length": 0,
            "tail_length": 30,
            "tail_width": 2,
        },
        # Additional attributes for BoxesStyle
        BoxesStyle: {
            "edge_width": 3,
            "opacity": 1.0,
            "shape_type": "rectangle",
            "face_color": "#FFFFFF00",
            "edge_colormap": DEFAULT_COLORMAP,
            "text": {
                "visible": True,
                "anchor": "lower_left",
                "translation": 5,
            },
        },
        # Additional attributes for RoiStyle
        RoisStyle: {
            "color": "red",
            "edge_width": 5.0,
            "opacity": 1.0,
            "text": {
                "visible": True,
                "anchor": "center",
            },
        },
    }


@pytest.mark.parametrize(
    "layer_class",
    [LayerStyle, PointsStyle, TracksStyle, BoxesStyle, RoisStyle],
)
def test_layer_style_initialization(
    sample_layer_style, layer_class, default_style_attributes
):
    """Test that LayerStyle and subclasses initialize with default values."""
    style = sample_layer_style(layer_class)

    # Expected attributes of base LayerStyle, shared by all subclasses
    expected_attrs = default_style_attributes[LayerStyle].copy()

    # Additional attributes, specific to subclasses of LayerStyle
    if layer_class != LayerStyle:
        expected_attrs.update(default_style_attributes[layer_class])

    # Check that all attributes are set correctly
    for attr, expected_value in expected_attrs.items():
        actual_value = getattr(style, attr)
        if isinstance(expected_value, pd.DataFrame):
            assert actual_value.equals(expected_value)
        else:
            assert actual_value == expected_value


def test_layer_style_as_kwargs(sample_layer_style, default_style_attributes):
    """Test that the as_kwargs method returns the correct dictionary."""
    style = sample_layer_style(LayerStyle).as_kwargs()
    expected_attrs = default_style_attributes[LayerStyle]
    assert style == expected_attrs


@pytest.mark.parametrize(
    "property, expected_n_colors",
    [
        ("category", 3),
        ("value", 5),
    ],
)
@pytest.mark.parametrize(
    "points_style_text_dict",
    [
        "default",
        "with_color_key",
    ],
)
def test_points_style_set_color_by(
    sample_layer_style,
    sample_properties,
    points_style_text_dict,
    property,
    expected_n_colors,
):
    """Test that set_color_by updates the color and color cycle of
    the point markers and the text.
    """
    # Create a points style object with predefined properties
    points_style = sample_layer_style(PointsStyle)

    # Add a color key to the text dictionary if required
    if points_style_text_dict == "with_color_key":
        points_style.text = {"color": {"fallback": "white"}}

    # Color markers and text by the property "prop"
    points_style.set_color_by(
        property=property,
        properties_df=sample_properties,
    )

    # Check that the markers and the text color follow "prop"
    assert points_style.face_color == property
    assert "color" in points_style.text
    assert points_style.text["color"]["feature"] == property

    # Check the color cycle
    color_cycle = _sample_colormap(
        len(sample_properties[property].unique()),
        cmap_name=DEFAULT_COLORMAP,
    )
    assert points_style.face_color_cycle == color_cycle
    assert points_style.text["color"]["colormap"] == color_cycle

    # Check number of colors is as expected
    assert len(points_style.face_color_cycle) == expected_n_colors
    assert len(points_style.text["color"]["colormap"]) == expected_n_colors

    # Check that all colors are tuples of length 4 (RGBA)
    assert all(
        isinstance(c, tuple) and len(c) == 4
        for c in points_style.face_color_cycle
    )


@pytest.mark.parametrize(
    "property",
    [
        "category",
        "value",
    ],
)
@pytest.mark.parametrize(
    "layer_type",
    [
        PointsStyle,
        BoxesStyle,
    ],
)
def test_layer_style_set_text_by(
    property, layer_type, sample_layer_style, default_style_attributes
):
    """Test that set_text_by updates the text property of the points layer."""
    # Create a points style object with predefined properties
    layer_style = sample_layer_style(layer_type)

    # Get the default attributes
    default_layer_style = default_style_attributes[layer_type]

    # Check there is no text set
    assert (
        "string" not in layer_style.text
        or layer_style.text["string"] != property
    )

    # Set text by the input property
    layer_style.set_text_by(property=property)

    # Check that the text properties are as expected
    assert layer_style.text["string"] == property
    assert all(
        layer_style.text[attr] == default_layer_style["text"][attr]
        for attr in default_layer_style["text"]
    )


@pytest.mark.parametrize(
    "set_color_by_kwargs",
    [
        {"property": "category", "cmap": None},
        {"property": "category", "cmap": "viridis"},
        {"property": "value", "cmap": None},
        {"property": "value", "cmap": "viridis"},
    ],
)
def test_tracks_style_color_by(
    set_color_by_kwargs, sample_layer_style, default_style_attributes
):
    """Test that set_color_by updates the color of the tracks layer."""
    # Create a tracks style object with predefined properties
    tracks_style = sample_layer_style(TracksStyle)

    # Get the default attributes
    default_tracks_style = default_style_attributes[TracksStyle]

    # Check the default colormap and color_by properties
    assert tracks_style.colormap == default_tracks_style["colormap"]
    assert tracks_style.color_by == default_tracks_style["color_by"]

    # Set color by the property
    tracks_style.set_color_by(**set_color_by_kwargs)

    # Check that the color_by property is set correctly
    assert tracks_style.color_by == set_color_by_kwargs["property"]

    # Check that the colormap is set correctly
    if set_color_by_kwargs["cmap"] is None:
        assert tracks_style.colormap == default_tracks_style["colormap"]
    else:
        assert tracks_style.colormap == set_color_by_kwargs["cmap"]


@pytest.mark.parametrize(
    "color_property, n_unique_values",
    [
        ("property_1", 1),
        ("property_2", 2),
    ],
)
def test_boxes_style_set_color_by(
    color_property,
    n_unique_values,
    sample_layer_style,
    sample_properties_with_factorized,
):
    """Test that set_color_by updates the color and color cycle of
    the bounding boxes and the text.
    """
    # Create a shapes style object with predefined properties
    boxes_style = sample_layer_style(BoxesStyle)

    # Create a properties dataframe with the input property and a factorized
    # version of the same property
    properties_df = sample_properties_with_factorized(
        color_property, n_unique_values
    )

    # Set text and edge color to follow the input property
    boxes_style.set_color_by(
        property=color_property,
        properties_df=properties_df,
    )

    # Generate the color cycle for the factorized property
    color_property_factorized = color_property + "_factorized"
    n_colors = len(properties_df[color_property_factorized].unique())
    color_cycle = _sample_colormap(n_colors, cmap_name=DEFAULT_COLORMAP)

    # Check that the bboxes edges and text colormaps match the computed
    # color cycle
    assert boxes_style.edge_color_cycle == color_cycle
    assert boxes_style.text["color"]["colormap"] == color_cycle

    # Check that the number of colors matches the number of unique values
    # in the input property
    assert len(boxes_style.edge_color_cycle) == n_unique_values
    assert len(boxes_style.text["color"]["colormap"]) == n_unique_values

    # Check that all colors are tuples of length 4 (RGBA)
    assert all(
        isinstance(c, tuple) and len(c) == 4
        for c in boxes_style.edge_color_cycle
    )


@pytest.mark.parametrize(
    ["color", "expected_rgb"],
    [
        pytest.param("blue", (0.0, 0.0, 1.0), id="blue_as_str"),
        pytest.param("red", (1.0, 0.0, 0.0), id="red_as_str"),
        pytest.param((1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0), id="red_as_tuple"),
        pytest.param(
            (0.0, 0.0, 1.0, 0.5), (0.0, 0.0, 1.0), id="blue_as_tuple_alpha"
        ),
    ],
)
def test_rois_style_colors(color, expected_rgb):
    """Test that setting the color attribute updates the face, edge,
    and text colors. The face color must be transparent, while edges and
    text must be opaque.
    """
    # Create a ROIs style object
    rois_style = RoisStyle()
    rois_style.color = color

    # Convert expected_rgb to RGBA for comparison
    expected_rgba = expected_rgb + (1.0,)
    expected_face_rgba = expected_rgb + (0.25,)

    assert_array_equal(rois_style.edge_and_text_color, expected_rgba)
    assert_array_equal(rois_style.face_color, expected_face_rgba)
