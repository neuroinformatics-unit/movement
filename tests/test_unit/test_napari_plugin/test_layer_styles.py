"""Unit tests for the LayerStyle and PointsStyle classes."""

import pandas as pd
import pytest

from movement.napari.layer_styles import (
    DEFAULT_COLORMAP,
    LayerStyle,
    PointsStyle,
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
            "text": {"visible": False},
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
def test_points_style_set_color_by(
    sample_layer_style, prop, expected_n_colors
):
    """Test that set_color_by updates face_color and face_color_cycle."""
    points_style = sample_layer_style(PointsStyle)

    points_style.set_color_by(prop=prop)
    # Check that face_color and text are updated correctly
    assert points_style.face_color == prop
    assert points_style.text == {"visible": False, "string": prop}

    # Check that face_color_cycle has the correct number of colors
    assert len(points_style.face_color_cycle) == expected_n_colors
    # Check that all colors are tuples of length 4 (RGBA)
    assert all(
        isinstance(c, tuple) and len(c) == 4
        for c in points_style.face_color_cycle
    )
