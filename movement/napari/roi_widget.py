"""Widget for defining regions of interest (ROIs).

ROIs are drawn as shapes in a napari Shapes layer
and shown in a table view.

See the `Qt Model/View framework
<https://doc.qt.io/qt-6/model-view-programming.html>`_
for more background on this widget's architecture.
"""

from contextlib import suppress

from napari.layers import Shapes
from napari.viewer import Viewer
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class RoiTableModel(QAbstractTableModel):
    """Table model for napari Shapes layer ROIs."""

    def __init__(self, shapes_layer: Shapes, parent=None):
        """Initialize the ROI model with a Shapes layer."""
        super().__init__(parent)
        self.layer = shapes_layer
        # Connect to layer events
        self.layer.events.data.connect(self._on_layer_data_changed)

    def rowCount(self, parent=QModelIndex()):  # noqa: B008
        """Return the number of ROIs in the Shapes layer."""
        return len(self.layer.data) if self.layer else 0

    def columnCount(self, parent=QModelIndex()):  # noqa: B008
        """Return the number of columns in the ROI table."""
        return 2 if self.layer else 0

    def data(self, index, role=Qt.DisplayRole):
        """Return the actual data to be shown in each cell of the table."""
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        row, col = index.row(), index.column()

        if row >= len(self.layer.data):
            return None

        if col == 0:
            names = self.layer.properties.get("name", [])
            return names[row] if row < len(names) else ""
        elif col == 1:
            return (
                self.layer.shape_type[row]
                if row < len(self.layer.shape_type)
                else ""
            )
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Supply the column names for the table."""
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return ["Name", "Shape type"][section]
        else:  # Vertical orientation
            return str(section)  # Return the row index as a string

    def _on_layer_data_changed(self, event=None):
        """Update the model when the ROI Shapes layer data changes."""
        if self.layer is None or event.action not in ["added", "removed"]:
            return

        # Note that this list includes the just added shapes (if any),
        # but this could be a duplicate of an existing name.
        current_names = [
            n
            for n in self.layer.properties.get("name", [])
            if isinstance(n, str)
        ]

        # This ensures new ROIs are given unique names.
        updated_names = (
            self._update_roi_names(current_names)
            if event.action == "added"
            else current_names  # No need to update names on shape removal
        )

        self.layer.properties = {"name": updated_names}

        self.layer.text = {
            "string": "{name}",
            "color": "white",
        }

        self.beginResetModel()
        self.endResetModel()

    def _on_layer_deleted(self, event=None):
        """Handle the deletion of the ROI Shapes layer."""
        # Only reset the model if the layer being removed
        # is the one we are currently using.
        if event.value == self.layer:
            self.layer.events.data.disconnect(self._on_layer_data_changed)
            self.layer = None
            self.beginResetModel()
            self.endResetModel()

    def _update_roi_names(self, existing_names: list) -> list:
        """Update the names of existing ROIs.

        We name ROIs in the format "ROI-<number>". The number is
        incremented based on existing ROIs with such auto-assigned names.
        """
        updated_names = existing_names.copy()

        # Find the maximum number of existing ROIs with auto-assigned names
        auto_names = [
            name for name in existing_names if name.startswith("ROI-")
        ]

        if auto_names:
            auto_numbers = []
            for name in auto_names:
                # Skip names that don't follow ROI-<number> pattern
                with suppress(ValueError):
                    auto_numbers.append(int(name.split("-")[-1]))
            max_number = max(auto_numbers) if auto_numbers else 0
        else:
            max_number = 0

        # Assign the next available name
        next_auto_name = f"ROI-{max_number + 1}"
        if existing_names:  # to the last existing ROI
            updated_names[-1] = next_auto_name
        else:  # No existing ROIs, so add the first one
            updated_names.append(next_auto_name)
        return updated_names


class RoiTableView(QTableView):
    """Table view for displaying and managing ROIs."""

    def __init__(self, parent=None):
        """Initialize the ROI table view."""
        super().__init__(parent=parent)
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)
        self.current_model: RoiTableModel | None = None

    def setModel(self, model):
        """Override setModel to connect selection signals."""
        super().setModel(model)
        self.current_model = model

        if model is not None:
            self.selectionModel().selectionChanged.connect(
                self._on_selection_changed
            )

    def _on_selection_changed(self, selected, deselected):
        """Handle table row selection changes."""
        if self.current_model is None or self.current_model.layer is None:
            return

        # Get the selected row index
        indexes = selected.indexes()
        if not indexes:
            return

        # Select the corresponding shape in napari
        row = indexes[0].row()
        if row < len(self.current_model.layer.data):
            self.current_model.layer.selected_data = {row}


class RoiWidget(QWidget):
    """Widget for defining regions of interest (ROIs).

    ROIs are drawn as shapes in a napari Shapes layer
    and shown in a table view
    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the ROI widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.current_model: RoiTableModel | None = None

        self._setup_ui()
        self._connect_signals()
        self._update_layer_dropdown()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create layer selection group box
        layer_group = QGroupBox("Select a shapes layer to draw ROIs")
        layer_group.setToolTip(
            "Select an existing shapes layer to draw ROIs on, "
            "or create a new one. Only shapes layers that start with "
            "'ROI' are considered."
        )
        layer_layout = self._create_layer_selection_layout()
        layer_group.setLayout(layer_layout)
        main_layout.addWidget(layer_group)

        # Create table view group box
        table_group = QGroupBox("ROIs drawn in this layer")
        table_group.setToolTip(
            "Use napari layer controls (top left) to draw shapes."
        )
        table_layout = QVBoxLayout()
        self.roi_table_view = RoiTableView(self)
        table_layout.addWidget(self.roi_table_view)
        table_group.setLayout(table_layout)
        main_layout.addWidget(table_group)

    def _connect_signals(self):
        """Connect layer events to update dropdown."""
        self.viewer.layers.events.inserted.connect(self._update_layer_dropdown)
        self.viewer.layers.events.removed.connect(self._update_layer_dropdown)

    def _create_layer_selection_layout(self):
        """Create the ROI layer selection layout with dropdown and button."""
        layer_picker_layout = QHBoxLayout()

        # Layer selection dropdown
        self.layer_picker = QComboBox()
        self.layer_picker.setMinimumWidth(150)
        self.layer_picker.currentTextChanged.connect(self._on_layer_selected)

        # Create new layer button
        self.create_layer_button = QPushButton("Create new ROI layer")
        self.create_layer_button.clicked.connect(self._create_new_layer)

        layer_picker_layout.addWidget(self.layer_picker)
        layer_picker_layout.addWidget(self.create_layer_button)
        layer_picker_layout.addStretch()  # Push widgets to the left

        return layer_picker_layout

    def _get_roi_layers(self) -> dict[str, Shapes]:
        """Get all ROI layers (Shapes layers that start with 'ROI').

        Returns a dictionary with layer names as keys and layers as values.
        """
        return {
            layer.name: layer
            for layer in self.viewer.layers
            if isinstance(layer, Shapes) and layer.name.startswith("ROI")
        }

    def _get_roi_layer_by_name(self, name: str) -> Shapes | None:
        """Return the ROI layer with the given name."""
        return self._get_roi_layers().get(name)

    def _get_roi_layer_names(self) -> list[str]:
        """Return the names of all ROI layers."""
        return list(self._get_roi_layers().keys())

    def _update_layer_dropdown(self, event=None):
        """Update the layer dropdown with current ROI layers.

        We consider only Shapes layers that start with "ROI".
        """
        current_text = self.layer_picker.currentText()
        roi_layer_names = self._get_roi_layer_names()

        self.layer_picker.clear()
        if roi_layer_names:
            self.layer_picker.addItems(roi_layer_names)
            if current_text in roi_layer_names:
                # Try to restore previous selection
                self.layer_picker.setCurrentText(current_text)
            else:
                self.layer_picker.setCurrentIndex(0)
        else:
            self.layer_picker.addItem("No ROI layers available")

        # Update button state
        self.create_layer_button.setEnabled(True)

    def _on_layer_selected(self, layer_name: str):
        """Handle layer selection from dropdown."""
        if not layer_name or layer_name == "No ROI layers available":
            self._clear_table_model()
            self.viewer.layers.selection.clear()
            return

        roi_layer = self._get_roi_layer_by_name(layer_name)
        if roi_layer is not None:
            # Select the layer in napari
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(roi_layer)
            # Connect the layer to the table model
            self._link_layer_to_model(roi_layer)

    def _create_new_layer(self):
        """Create a new ROI layer and select it."""
        # Let napari handle the naming automatically
        new_layer = self.viewer.add_shapes(name="ROIs")
        # Select the new layer in the layer picker
        self.layer_picker.setCurrentText(new_layer.name)

    def _link_layer_to_model(self, roi_layer: Shapes):
        """Link a ROI layer to a table model."""
        # Disconnect previous model if it exists
        if self.current_model is not None:
            self.current_model.layer.events.data.disconnect(
                self.current_model._on_layer_data_changed
            )
            self.current_model.layer.events.name.disconnect(
                self._on_layer_renamed
            )
            self.viewer.layers.events.removed.disconnect(
                self.current_model._on_layer_deleted
            )

        # Create new model
        self.current_model = RoiTableModel(roi_layer)
        self.roi_table_view.setModel(self.current_model)

        # Connect layer events
        self.viewer.layers.events.removed.connect(
            self.current_model._on_layer_deleted
        )
        # Connect to layer name changes
        roi_layer.events.name.connect(self._on_layer_renamed)

    def _on_layer_renamed(self, event=None):
        """Handle layer renaming by updating the dropdown."""
        # Update the dropdown to reflect the new name
        self._update_layer_dropdown()

        # Automatically select the renamed layer in the picker
        self.layer_picker.setCurrentText(event.source.name)

    def _clear_table_model(self):
        """Clear the table model when no layer is selected."""
        if self.current_model is not None:
            self.current_model.layer.events.data.disconnect(
                self.current_model._on_layer_data_changed
            )
            self.current_model.layer.events.name.disconnect(
                self._on_layer_renamed
            )
            self.viewer.layers.events.removed.disconnect(
                self.current_model._on_layer_deleted
            )
            self.current_model = None

        self.roi_table_view.setModel(None)
