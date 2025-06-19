import numpy as np
import pytest

from movement.napari.export_widgets import ShapesExporter
from movement.roi import LineOfInterest, PolygonOfInterest


class MockEvents:
    def __init__(self):
        self.inserted = MockSignal()
        self.removed = MockSignal()


class MockSignal:
    def connect(self, callback):
        pass  # No-op for testing


class MockLayers(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = MockEvents()


@pytest.fixture
def mock_shapes_data():
    """Create sample shapes data."""
    return {
        "data": [
            np.array([[0, 0], [1, 1]]),  # Line
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # Polygon
        ],
        "shape_type": ["line", "polygon"],
        "name": "test_shapes",
    }


@pytest.fixture
def mock_viewer(monkeypatch):
    """Create a mock viewer with basic required functionality."""

    class MockViewer:
        def __init__(self):
            self.layers = MockLayers()

        def add_shapes(self, *args, **kwargs):
            return type(
                "MockLayer",
                (),
                {
                    "shape_type": kwargs.get("shape_type", []),
                    "properties": kwargs.get("properties", {}),
                    "name": kwargs.get("name", "test"),
                },
            )()

    return MockViewer()


@pytest.fixture
def exporter(mock_viewer, monkeypatch):
    """Create ShapesExporter with mocked components."""
    # Avoid GUI prompts
    monkeypatch.setattr(
        "qtpy.QtWidgets.QInputDialog.getText",
        lambda *args, **kwargs: ("test_roi", True),
    )
    return ShapesExporter(mock_viewer)


@pytest.fixture
def mock_shapes_layer():
    """Create a mock shapes layer with required attributes."""

    class ShapesLayer:
        def __init__(self):
            self.name = "test_shapes"
            self.__class__ = type("Shapes", (), {"__name__": "Shapes"})

    return ShapesLayer()


def test_convert_line_to_roi(exporter, mock_shapes_data):
    """Test converting line shape to ROI."""
    # Create minimal shapes layer with line data
    shapes = type(
        "ShapesLayer",
        (),
        {
            "data": [mock_shapes_data["data"][0]],
            "shape_type": ["line"],
            "__class__": type("Shapes", (), {"__name__": "Shapes"}),
        },
    )

    roi_objects = exporter._convert_shapes_to_roi(shapes)
    assert len(roi_objects) == 1
    assert isinstance(roi_objects[0], LineOfInterest)


def test_convert_polygon_to_roi(exporter, mock_shapes_data):
    """Test converting polygon shape to ROI."""
    shapes = type(
        "ShapesLayer",
        (),
        {
            "data": [mock_shapes_data["data"][1]],
            "shape_type": ["polygon"],
            "__class__": type("Shapes", (), {"__name__": "Shapes"}),
        },
    )

    roi_objects = exporter._convert_shapes_to_roi(shapes)
    assert len(roi_objects) == 1
    assert isinstance(roi_objects[0], PolygonOfInterest)


def test_export_and_load_roi(exporter, tmp_path):
    """Test round trip of exporting and loading ROIs."""
    # Create test ROIs
    line = LineOfInterest(np.array([[0, 0], [1, 1]]), name="test_line")
    polygon = PolygonOfInterest(
        exterior_boundary=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        name="test_polygon",
    )
    original_rois = [line, polygon]

    # Export
    file_path = tmp_path / "test_rois.json"
    exporter._export_roi_objects(original_rois, file_path)

    # Load and verify
    loaded_rois = exporter._load_roi_objects(file_path)
    assert len(loaded_rois) == len(original_rois)
    assert loaded_rois[0].name == "test_line"
    assert loaded_rois[1].name == "test_polygon"


def test_create_roi_layer(exporter, mock_shapes_data):
    """Test creating napari layer from ROIs."""
    # Create sample ROIs
    rois = [
        LineOfInterest(np.array([[0, 0], [1, 1]]), name="test_line"),
        PolygonOfInterest(
            exterior_boundary=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            name="test_polygon",
        ),
    ]

    layer = exporter._create_roi_layer(rois)
    assert layer.shape_type == ["line", "polygon"]
    assert layer.properties["name"] == ["test_line", "test_polygon"]


def test_layer_list_update(exporter, mock_shapes_layer):
    """Test layer list updates with available shapes."""
    # Add the mock shapes layer to viewer
    exporter.viewer.layers.append(mock_shapes_layer)

    # Update the layer list
    exporter._update_layer_list()

    # Verify the combo box contents
    assert exporter.layer_combo.count() == 1
    assert exporter.layer_combo.currentText() == "test_shapes"
