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


class RoisWidget(QWidget):
    """Widget for defining regions of interest (ROIs).

    The widget provides a dropdown to select an existing ROIs layer, i.e.
    a Shapes layer whose name starts with "ROIs", and a button to add a new
    ROIs layer.

    The widget also provides a table view which displays the shapes drawn
    in the currently selected ROIs layer. Clicking on a row in the table
    view selects the corresponding shape in the ROIs layer. Shapes are
    auto-named in the format "ROI-<number>" (stored in the layer's
    text property), but this can be edited by double-clicking on the Name
    column of the table view.
    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the ROI widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.roi_table_model: RoisTableModel | None = None
        self.roi_table_view = RoisTableView(self)

        self._setup_ui()
        self._connect_signals()
        self._update_layer_dropdown()

    def _setup_ui(self):
        """Set up the user interface with two groupboxes.

        The first groupbox contains the layer controls:
        a dropdown to select an existing ROIs layer
        and a button to add a new ROIs layer.
        The second groupbox contains the table view.
        """
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create layer controls group box
        layer_controls_group = QGroupBox("Layer to draw ROIs on")
        layer_controls_group.setToolTip(
            "Select an existing shapes layer to draw ROIs on, "
            "or create a new one.\nOnly shapes layers that start with "
            "'ROI' are considered."
        )
        layer_controls_group.setLayout(self._setup_layer_controls_layout())
        main_layout.addWidget(layer_controls_group)

        # Create table view group box
        table_view_group = QGroupBox("ROIs drawn in this layer")
        table_view_group.setToolTip(
            "Use napari layer controls (top left) to draw shapes."
        )
        table_view_group.setLayout(self._setup_table_view_layout())
        main_layout.addWidget(table_view_group)

    def _setup_layer_controls_layout(self):
        """Create the ROIs layer controls layout with dropdown and button."""
        layer_controls_layout = QHBoxLayout()

        self.layer_dropdown = QComboBox()
        self.layer_dropdown.setMinimumWidth(150)
        self.layer_dropdown.currentTextChanged.connect(self._on_layer_selected)

        self.add_layer_button = QPushButton("Add new layer")
        self.add_layer_button.setEnabled(True)
        self.add_layer_button.clicked.connect(self._add_new_layer)

        layer_controls_layout.addWidget(self.layer_dropdown)
        layer_controls_layout.addWidget(self.add_layer_button)

        return layer_controls_layout

    def _setup_table_view_layout(self):
        """Create the ROI table view layout."""
        table_view_layout = QVBoxLayout()
        table_view_layout.addWidget(self.roi_table_view)
        return table_view_layout

    def _connect_signals(self):
        """Connect layer events to update dropdown."""
        self.viewer.layers.events.inserted.connect(self._update_layer_dropdown)
        self.viewer.layers.events.removed.connect(self._update_layer_dropdown)

        # Connect to name change events for all existing shapes layers
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                layer.events.name.connect(self._update_layer_dropdown)

    def _get_roi_layers(self) -> dict[str, Shapes]:
        """Get all ROIs layers (Shapes layers that start with 'ROI').

        Returns a dictionary with layer names as keys and layers as values.
        """
        return {
            layer.name: layer
            for layer in self.viewer.layers
            if isinstance(layer, Shapes) and layer.name.startswith("ROI")
        }

    def _update_layer_dropdown(self, event=None):
        """Update the layer dropdown with current ROIs layers."""
        # Connect to name change events for any new Shapes layers
        if event is not None and hasattr(event, "value"):
            layer = event.value
            if isinstance(layer, Shapes):
                layer.events.name.connect(self._update_layer_dropdown)

        current_text = self.layer_dropdown.currentText()
        roi_layer_names = list(self._get_roi_layers().keys())

        self.layer_dropdown.clear()
        if roi_layer_names:
            self.layer_dropdown.addItems(roi_layer_names)
            renamed_to_roi = (  # True if a layer was renamed "ROI*"
                event is not None
                and hasattr(event, "source")
                and isinstance(event.source, Shapes)
                and event.source.name.startswith("ROI")
            )
            # Determine which layer to select
            if renamed_to_roi:
                # Auto-select the newly renamed ROI layer
                self.layer_dropdown.setCurrentText(event.source.name)
            elif current_text in roi_layer_names:
                # Next, try restoring the previous selection
                self.layer_dropdown.setCurrentText(current_text)
            else:
                # Fall back to the first layer
                self.layer_dropdown.setCurrentIndex(0)
        else:
            self.layer_dropdown.addItem("Select a layer")

    def _on_layer_selected(self, layer_name: str):
        """Handle layer selection from dropdown."""
        if not layer_name or layer_name == "Select a layer":
            self._clear_roi_table_model()
            self.viewer.layers.selection.clear()
            return

        roi_layer = self._get_roi_layers().get(layer_name)
        if roi_layer is not None:
            # Select the layer in napari
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(roi_layer)
            # Connect the ROIs layer to the table model
            self._link_layer_to_model(roi_layer)

    def _add_new_layer(self):
        """Create a new ROIs layer and select it."""
        new_layer = self.viewer.add_shapes(name="ROIs")
        self.layer_dropdown.setCurrentText(new_layer.name)

    def _link_layer_to_model(self, roi_layer: Shapes):
        """Link an ROIs layer to an ROIs table model."""
        # Disconnect previous model if it exists
        self._disconnect_table_model_signals()

        # Auto-assign names if the layer has shapes without names.
        self._auto_assign_roi_names(roi_layer)

        # Create new model and link it to the table view
        self.roi_table_model = RoisTableModel(roi_layer)
        self.roi_table_view.setModel(self.roi_table_model)

        # The model will listen to napari layer removal events
        self.viewer.layers.events.removed.connect(
            self.roi_table_model._on_layer_deleted
        )
        # Connect to layer name changes
        roi_layer.events.name.connect(self._on_layer_renamed)

    def _auto_assign_roi_names(self, roi_layer: Shapes) -> None:
        """Auto-assign names to ROIs if the layer has shapes without names.

        This happens if shapes are drawn while the layer's name does not
        start with "ROI".
        """
        if len(roi_layer.data) > 0 and "name" not in roi_layer.properties:
            names = [f"ROI-{i + 1}" for i in range(len(roi_layer.data))]
            roi_layer.properties = {"name": names}
            roi_layer.text = {"string": "{name}", "color": "white"}

    def _on_layer_renamed(self, event=None):
        """Handle layer renaming by updating the dropdown."""
        self._update_layer_dropdown()
        self.layer_dropdown.setCurrentText(event.source.name)

    def _disconnect_table_model_signals(self):
        """Disconnect signals from the ROIs table model."""
        if self.roi_table_model is not None:
            self.roi_table_model.layer.events.data.disconnect(
                self.roi_table_model._on_layer_data_changed
            )
            self.roi_table_model.layer.events.name.disconnect(
                self._on_layer_renamed
            )
            self.viewer.layers.events.removed.disconnect(
                self.roi_table_model._on_layer_deleted
            )

    def _clear_roi_table_model(self):
        """Clear the ROIs table model."""
        self._disconnect_table_model_signals()
        self.roi_table_model = None
        self.roi_table_view.setModel(None)


class RoisTableView(QTableView):
    """Table view for displaying and managing ROIs."""

    def __init__(self, parent=None):
        """Initialize the ROI table view."""
        super().__init__(parent=parent)
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)
        self.setEditTriggers(
            QTableView.DoubleClicked | QTableView.EditKeyPressed
        )
        self.current_model: RoisTableModel | None = None

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


class RoisTableModel(QAbstractTableModel):
    """Table model for ROIs defined in a napari Shapes layer."""

    def __init__(self, shapes_layer: Shapes, parent=None):
        """Initialize the ROIs table model with a Shapes layer."""
        super().__init__(parent)
        self.layer = shapes_layer
        # The model will listen to napari layer data changes
        self.layer.events.data.connect(self._on_layer_data_changed)

    def rowCount(self, parent=QModelIndex()):  # noqa: B008
        """Match the number of ROIs in the Shapes layer."""
        return len(self.layer.data) if self.layer else 0

    def columnCount(self, parent=QModelIndex()):  # noqa: B008
        """Fix the number of columns in the ROIs table."""
        return 2 if self.layer else 0

    def data(self, index, role=Qt.DisplayRole):
        """Return the actual data to be shown in each cell of the table."""
        if not index.isValid():
            return None

        row, col = index.row(), index.column()

        if row >= len(self.layer.data):
            return None

        if role == Qt.DisplayRole:
            if col == 0:
                return self._get_roi_name_for_row(row)
            elif col == 1:
                return (
                    self.layer.shape_type[row]
                    if row < len(self.layer.shape_type)
                    else ""
                )
        elif role == Qt.EditRole and col == 0:
            # Return editable data for the Name column
            return self._get_roi_name_for_row(row)
        return None

    def flags(self, index):
        """Return the item flags for the given index."""
        if not index.isValid():
            return Qt.NoItemFlags

        if index.column() == 0:  # Make only the Name column editable
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def setData(self, index, value, role=Qt.EditRole):
        """Set the data for the given index.

        This allows the user to edit the name of the ROI.
        """
        if not index.isValid() or role != Qt.EditRole:
            return False

        row, col = index.row(), index.column()

        if row >= len(self.layer.data):
            return False

        # Only allow editing the Name column
        if col == 0:
            names = list(self.layer.properties.get("name", []))

            while len(names) <= row:
                names.append("")  # Ensure we have enough names
            names[row] = str(value)  # Update the name
            self.layer.properties = {"name": names}  # Update layer properties
            self.dataChanged.emit(index, index)
            return True

        return False

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Supply the column names for the table."""
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return ["Name", "Shape type"][section]
        else:  # Vertical orientation
            return str(section)  # Return the row index as a string

    def _get_roi_name_for_row(self, row):
        """Get the ROI name corresponding to a specific row."""
        names = self.layer.properties.get("name", [])
        return names[row] if row < len(names) else ""

    def _on_layer_data_changed(self, event=None):
        """Update the model when the ROIs Shapes layer data changes."""
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
        """Handle the deletion of the ROIs Shapes layer."""
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
