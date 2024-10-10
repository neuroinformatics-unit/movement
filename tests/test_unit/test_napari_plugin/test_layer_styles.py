"""Unit tests for the LayerStyle and PointsStyle classes."""

import pandas as pd
import pytest

from movement.napari.layer_styles import (
    DEFAULT_COLORMAP,
    LayerStyle,
    PointsStyle,
)


# Create a pytest fixture for the properties DataFrame
@pytest.fixture
def sample_properties():
    """Fixture that provides a sample DataFrame."""
    data = {"category": ["A", "B", "A", "C", "B"], "value": [1, 2, 3, 4, 5]}
    return pd.DataFrame(data)


# Test cases for LayerStyle
def test_layer_style_initialization(sample_properties):
    """Test that LayerStyle initializes with default values."""
    style = LayerStyle(name="Layer1", properties=sample_properties)

    assert style.name == "Layer1"
    assert style.visible is True
    assert style.blending == "translucent"
    assert style.properties.equals(sample_properties)


def test_layer_style_as_kwargs(sample_properties):
    """Test that the as_kwargs method returns the correct dictionary."""
    style = LayerStyle(name="Layer1", properties=sample_properties)

    expected_dict = {
        "name": "Layer1",
        "properties": sample_properties,
        "visible": True,
        "blending": "translucent",
    }

    assert style.as_kwargs() == expected_dict


# Test cases for PointsStyle
def test_points_style_initialization(sample_properties):
    """Test that PointsStyle initializes with correct default values."""
    points_style = PointsStyle(
        name="PointsLayer", properties=sample_properties
    )

    assert points_style.symbol == "disc"
    assert points_style.size == 10
    assert points_style.border_width == 0
    assert points_style.face_color is None
    assert points_style.face_color_cycle is None
    assert points_style.face_colormap == DEFAULT_COLORMAP
    assert points_style.text == {"visible": False}


@pytest.mark.parametrize(
    "prop, expected_n_colors",
    [
        ("category", 3),
        ("value", 5),
    ],
)
def test_points_style_set_color_by(sample_properties, prop, expected_n_colors):
    """Test the set_color_by method updates face_color and face_color_cycle.

    Also make sure that the number of colors in the face_color_cycle matches
    the number of unique values in the corresponding property column.
    """
    points_style = PointsStyle(
        name="PointsLayer", properties=sample_properties
    )

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
