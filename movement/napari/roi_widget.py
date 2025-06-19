"""Widget for defining regions of interest (ROIs).

Created Shapes are shown in a table view using the
[Qt Model/View framework](https://doc.qt.io/qt-6/model-view-programming.html)
"""

from napari.layers import Shapes
from napari.viewer import Viewer
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt
from qtpy.QtWidgets import (
    QFormLayout,
    QTableView,
    QWidget,
)


class RoiWidget(QWidget):
    """Widget for defining regions of interest (ROIs)."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the ROI widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())
        self.roi_table_view = RoiTableView(self)  # initially no model
        self.layout().addRow(self.roi_table_view)

    def ensure_initialised(self):
        """Ensure the Shapes layer and ROI table model are set up."""
        shapes_layer = self._get_or_create_shapes_layer("ROIs")
        current_model = self.roi_table_view.model()

        model_is_valid = (
            isinstance(current_model, RoiTableModel)
            and current_model.layer in self.viewer.layers
        )

        if not model_is_valid:
            model = RoiTableModel(shapes_layer)
            self.roi_table_view.setModel(model)
            # When a layer is removed, update the model
            self.viewer.layers.events.removed.connect(
                self.roi_table_view.model()._on_layer_deleted
            )

    def _get_shapes_layer(self, name: str) -> Shapes | None:
        """Return the Shapes layer with the given name, if it exists."""
        shape_layers = {
            layer.name: layer
            for layer in self.viewer.layers
            if isinstance(layer, Shapes)
        }
        return shape_layers.get(name)

    def _get_or_create_shapes_layer(self, name: str) -> Shapes:
        """Get the Shapes layer by name, or create one if it doesn't exist."""
        existing_layer = self._get_shapes_layer(name)
        if existing_layer is not None:
            return existing_layer

        return self.viewer.add_shapes(name=name)


class RoiTableView(QTableView):
    """Table view for displaying and managing ROIs."""

    def __init__(self, viewer: Viewer, parent=None):
        """Initialize the ROI table view."""
        super().__init__(parent=parent)
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)


class RoiTableModel(QAbstractTableModel):
    """Table model for napari Shapes layer ROIs."""

    def __init__(self, shapes_layer: Shapes, parent=None):
        """Initialize the ROI model with a Shapes layer."""
        super().__init__(parent)
        self.layer = shapes_layer
        # Connect to layer events
        self.layer.events.data.connect(self._on_layer_data_changed)

    def rowCount(self, parent=QModelIndex()):
        """Return the number of ROIs in the Shapes layer."""
        return len(self.layer.data) if self.layer else 0

    def columnCount(self, parent=QModelIndex()):
        """Return the number of columns in the ROI table."""
        return 2 if self.layer else 0

    def data(self, index, role=Qt.DisplayRole):
        """Return the actual data to be shown in each cell of the table."""
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        row, col = index.row(), index.column()
        if col == 0:
            return self.layer.properties.get("name", [""])[row]
        elif col == 1:
            return self.layer.shape_type[row]
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Supply the header labels for the table."""
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return ["ROI Name", "Type"][section]
        else:  # Vertical orientation
            return str(section)  # Return the row index as a string

    def _on_layer_data_changed(self, event=None):
        """Update the model when the Shapes layer data changes."""
        if self.layer is None or event.action not in ["added", "removed"]:
            return

        # Note that napari auto-extends the name property when new shapes
        # are added, so this list includes the just added shapes (if any).
        existing_names = [
            n
            for n in list(self.layer.properties.get("name", []))
            if isinstance(n, str)
        ]

        if event.action == "added":
            self._auto_assign_roi_names(existing_names)

        self.layer.properties = {"name": existing_names}

        self.layer.text = {
            "string": "{name}",
            "color": "white",
        }

        self.beginResetModel()
        self.endResetModel()

    def _on_layer_deleted(self, event=None):
        """Handle the deletion of the Shapes layer."""
        # Only reset the model if the layer being removed
        # is the one we are currently using.
        if event.value == self.layer:
            self.layer.events.data.disconnect(self._on_layer_data_changed)
            self.layer = None
            self.beginResetModel()
            self.endResetModel()

    def _auto_assign_roi_names(self, existing_names):
        """Automatically assign names to ROIs.

        We name ROIs in the format "ROI-<number>". The number is
        incremented based on the highest existing number among existing
        ROIs with such automatic names.
        """
        auto_names = [
            name for name in existing_names if name.startswith("ROI-")
        ]
        max_suffix = max(
            [int(name.split("-")[-1]) for name in auto_names] + [-1]
        )
        next_auto_name = f"ROI-{max_suffix + 1}"
        if existing_names:
            existing_names[-1] = next_auto_name
        else:
            existing_names.append(next_auto_name)
