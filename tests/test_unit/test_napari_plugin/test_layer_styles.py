"""Unit tests for the LayerStyle and PointsStyle classes."""

import pandas as pd
import pytest

from movement.napari.layer_styles import (
    DEFAULT_COLORMAP,
    LayerStyle,
    PointsStyle,
    _sample_colormap,
)


@pytest.fixture
def sample_properties():
    """Fixture that provides a sample "properties" DataFrame."""
    data = {"category": ["A", "B", "A", "C", "B"], "value": [1, 2, 3, 4, 5]}
    return pd.DataFrame(data)


@pytest.fixture
def sample_layer_style(sample_properties):
    """Fixture that provides a sample LayerStyle or subclass instance."""

    def _sample_layer_style(layer_class):
        return layer_class(name="Layer1", properties=sample_properties)

    return _sample_layer_style


@pytest.fixture
def default_style_attributes(sample_properties):
    """Fixture that provides expected attributes for LayerStyle and subclasses.

    It holds the default values we expect after initialisation, as well as the
    "name" and "properties" attributes that are defined in this test module.
    """
    return {
        # Shared attributes for LayerStyle and all its subclasses
        LayerStyle: {
            "name": "Layer1",  # as given in sample_layer_style
            "visible": True,
            "blending": "translucent",
            "properties": sample_properties,  # as given by fixture above
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
    }


@pytest.mark.parametrize(
    "layer_class",
    [LayerStyle, PointsStyle],
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
    "prop, expected_n_colors",
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
    sample_layer_style, points_style_text_dict, prop, expected_n_colors
):
    """Test that set_color_by updates the color and color cycle of
    the point markers and the text.
    """
    # Create a points style object with predefined properties
    points_style = sample_layer_style(PointsStyle)

    # add a color key to the text dictionary if required
    if points_style_text_dict == "with_color_key":
        points_style.text = {"color": {"fallback": "white"}}

    # Color markers and text by the property "prop"
    points_style.set_color_by(prop=prop)

    # Check that the markers and the text color follow "prop"
    assert points_style.face_color == prop
    assert "color" in points_style.text
    assert points_style.text["color"]["feature"] == prop

    # Check the color cycle
    color_cycle = _sample_colormap(
        len(points_style.properties[prop].unique()),
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
def test_points_style_set_text_by(
    property, sample_layer_style, default_style_attributes
):
    """Test that set_text_by updates the text property of the points layer."""
    # Create a points style object with predefined properties
    points_style = sample_layer_style(PointsStyle)

    # Get the default attributes
    default_points_style = default_style_attributes[PointsStyle]

    # Check there is no text set
    assert (
        "string" not in points_style.text
        or points_style.text["string"] != property
    )

    # Set text by the property "category"
    points_style.set_text_by(prop=property)

    # Check that the text properties are as expected
    assert points_style.text["string"] == property
    assert all(
        points_style.text[attr] == default_points_style["text"][attr]
        for attr in default_points_style["text"]
    )
