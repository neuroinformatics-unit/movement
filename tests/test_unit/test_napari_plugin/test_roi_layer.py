from unittest.mock import MagicMock

import numpy as np
import pytest
from napari.layers import Shapes

from movement.napari.roi_widget import ROIDrawingWidget


@pytest.fixture
def viewer_with_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ROIDrawingWidget(viewer)
    return viewer, widget


def test_create_roi_layer(viewer_with_widget):
    viewer, widget = viewer_with_widget
    assert widget.roi_layer is None

    # First creation
    widget._create_roi_layer()
    assert isinstance(widget.roi_layer, Shapes)
    assert widget.roi_layer.mode == "add_rectangle"
    assert "ROIs" in viewer.layers

    # Test duplicate creation
    widget._create_roi_layer()
    assert widget.status_label.text() == "ROI layer already exists"


def test_roi_selection(viewer_with_widget):
    viewer, widget = viewer_with_widget
    widget._create_roi_layer()

    # Add proper rectangle (4 points)
    test_rect = np.array([[10, 10], [10, 20], [30, 20], [30, 10]])
    widget.roi_layer.add(test_rect)

    # Simulate right-click inside ROI
    mock_event = MagicMock()
    mock_event.button = 2  # Right click
    mock_event.position = [15, 15]  # Inside rectangle
    mock_event.dims_displayed = [0, 1]  # Required for coordinate conversion

    widget._on_mouse_press(widget.roi_layer, mock_event)
    assert widget._selected_roi == 0
    assert widget.status_label.text() == "Selected ROI 1"

    # Test click outside ROI
    mock_event.position = [50, 50]
    widget._on_mouse_press(widget.roi_layer, mock_event)
    assert widget._selected_roi is None


@pytest.mark.parametrize(
    "point,rectangle,expected",
    [
        ((15, 15), [[10, 10], [30, 30]], True),
        ((10, 10), [[10, 10], [30, 30]], True),
        ((5, 5), [[10, 10], [30, 30]], False),
        ((35, 15), [[10, 10], [30, 30]], False),
    ],
)
def test_point_in_rectangle(point, rectangle, expected):
    rect = np.array(rectangle)
    result = ROIDrawingWidget._point_in_rectangle(np.array(point), rect)
    assert result == expected


def test_clear_rois(viewer_with_widget):
    viewer, widget = viewer_with_widget
    widget._create_roi_layer()

    # Add proper rectangle data
    widget.roi_layer.add(np.array([[0, 0], [0, 10], [10, 10], [10, 0]]))
    widget._clear_rois()

    assert len(widget.roi_layer.data) == 0
    assert widget.status_label.text() == "All ROIs cleared"


def test_roi_count_updates(viewer_with_widget):
    viewer, widget = viewer_with_widget
    widget._create_roi_layer()

    # Add first ROI
    widget.roi_layer.add(np.array([[0, 0], [0, 5], [5, 5], [5, 0]]))
    assert widget.status_label.text() == "Total ROIs: 1"

    # Add second ROI
    widget.roi_layer.add(np.array([[10, 10], [10, 15], [15, 15], [15, 10]]))
    assert widget.status_label.text() == "Total ROIs: 2"


def test_get_rois(viewer_with_widget):
    viewer, widget = viewer_with_widget
    widget._create_roi_layer()

    # Create proper rectangle data
    test_data = [
        np.array([[0, 0], [0, 5], [5, 5], [5, 0]]),
        np.array([[10, 10], [10, 15], [15, 15], [15, 10]]),
    ]
    widget.roi_layer.data = test_data

    # Verify ROI count
    assert len(widget.get_rois()) == 2
    assert widget.get_selected_roi() is None

    # Test selection comparison
    widget._selected_roi = 0
    selected = widget.get_selected_roi()
    assert selected is not None
    assert np.array_equal(selected, test_data[0]), (
        f"Expected:\n{test_data[0]}\nGot:\n{selected}"
    )
