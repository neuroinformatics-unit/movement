"""Widget for defining regions of interest (ROIs)."""

from napari.layers import Layer, Shapes
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)


class ROIWidget(QWidget):
    """Widget for defining regions of interest (ROIs)."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the ROI widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())

        # Create widgets
        self._create_roi_table()
        self._create_layer_selector()

    def _create_layer_selector(self):
        """Create a dropdown selecting a shapes layer."""
        self.layer_selector = QComboBox()
        self._update_layer_selector()
        self.layer_selector.currentIndexChanged.connect(
            self._on_layer_selected
        )
        self.layout().addRow("Shapes layer:", self.layer_selector)

        # Create a shapes layer if none exists
        if self.layer_selector.count() == 0:
            self._create_default_shapes_layer()

    def _create_roi_table(self):
        """Create the table for displaying ROIs."""
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(2)
        self.roi_table.setHorizontalHeaderLabels(["Name", "Type"])
        # Optional: make it read-only for now
        self.roi_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.roi_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.layout().addRow(self.roi_table)

    def _update_layer_selector(self):
        """Update the dropdown with current Shapes layers."""
        self.layer_selector.clear()
        shapes_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Shapes)
        ]
        for layer in shapes_layers:
            self.layer_selector.addItem(layer.name)

    def _get_layers_by_type(self, layer_type: Layer) -> list:
        """Return a list of napari layers of a given type."""
        return [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, layer_type)
        ]

    def _create_default_shapes_layer(self):
        """Create and select a default Shapes layer if none exists."""
        self.viewer.add_shapes(name="ROIs")
        self._update_layer_selector()
        index = self.layer_selector.findText("ROIs")
        self.layer_selector.setCurrentIndex(index)

    def _on_layer_selected(self, index):
        layer_name = self.layer_selector.currentText()
        self.shapes_layer = self.viewer.layers[layer_name]
        self._update_roi_table()
        self.shapes_layer.events.data.connect(self._update_roi_table)

    def _update_roi_table(self):
        """Update table based on current Shapes layer."""
        data = self.shapes_layer.data
        types = self.shapes_layer.shape_type

        self.roi_table.setRowCount(len(data))
        for i, (shape, shape_type) in enumerate(zip(data, types)):
            self.roi_table.setItem(i, 0, QTableWidgetItem(f"ROI {i}"))
            self.roi_table.setItem(i, 1, QTableWidgetItem(shape_type))
