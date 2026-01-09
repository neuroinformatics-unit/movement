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

from movement.napari.layer_styles import RoisColorManager, RoisStyle


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

    def __init__(self, napari_viewer: Viewer, cmap_name="tab10", parent=None):
        """Initialise the ROI widget.

        Parameters
        ----------
        napari_viewer : Viewer
            The napari viewer instance.
        cmap_name : str, optional
            Name of the napari colormap to use for ROI colors.
            Default is "tab10".
        parent : QWidget, optional
            The parent widget.

        """
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.color_manager = RoisColorManager(cmap_name=cmap_name)
        self.roi_table_model: RoisTableModel | None = None
        self.roi_table_view = RoisTableView(self)
        self._connected_layers: set[Shapes] = set()  # Track connected layers

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
            "or add a new one.\nOnly shapes layers that start with "
            "'ROI' are considered."
        )
        layer_controls_group.setLayout(self._setup_layer_controls_layout())
        main_layout.addWidget(layer_controls_group)

        # Create table view group box
        table_view_group = QGroupBox("ROIs drawn in this layer")
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
        self.viewer.layers.events.removed.connect(
            self._on_layer_removed_from_viewer
        )

        # Connect to name change events for all existing shapes layers
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                self._connect_layer_name_signal(layer)

    def _connect_layer_name_signal(self, layer: Shapes) -> None:
        """Connect to layer name change signal if not already connected."""
        if layer not in self._connected_layers:
            layer.events.name.connect(self._update_layer_dropdown)
            self._connected_layers.add(layer)

    def _disconnect_layer_name_signal(self, layer: Shapes) -> None:
        """Disconnect from layer name change signal."""
        if layer in self._connected_layers:
            layer.events.name.disconnect(self._update_layer_dropdown)
            self._connected_layers.discard(layer)

    def _is_roi_layer(self, layer: Shapes) -> bool:
        """Check if a Shapes layer is an ROI layer.

        First checks for explicit metadata marker, then falls back to
        case-insensitive name matching.
        """
        # Explicit metadata takes precedence
        if layer.metadata.get("movement_roi_layer", False):
            return True

        # Fall back to case-insensitive name heuristic
        return layer.name.upper().startswith("ROI")

    def _mark_as_roi_layer(self, layer: Shapes) -> None:
        """Mark a Shapes layer as an ROI layer via metadata."""
        layer.metadata["movement_roi_layer"] = True

    def _get_roi_layers(self) -> dict[str, Shapes]:
        """Get all ROIs layers.

        Returns a dictionary with layer names as keys and layers as values.
        """
        return {
            layer.name: layer
            for layer in self.viewer.layers
            if isinstance(layer, Shapes) and self._is_roi_layer(layer)
        }

    def _on_layer_removed_from_viewer(self, event=None):
        """Handle layer removal by disconnecting signals."""
        if event is not None and hasattr(event, "value"):
            layer = event.value
            if isinstance(layer, Shapes):
                self._disconnect_layer_name_signal(layer)
        self._update_layer_dropdown(event)

    def _update_layer_dropdown(self, event=None):
        """Update the layer dropdown with current ROIs layers."""
        # Connect to name change events for any new Shapes layers
        if event is not None and hasattr(event, "value"):
            layer = event.value
            if isinstance(layer, Shapes):
                self._connect_layer_name_signal(layer)

        # Check if a layer was renamed to an ROI name pattern
        # and mark it with metadata if so
        renamed_to_roi = False
        if (
            event is not None
            and hasattr(event, "source")
            and isinstance(event.source, Shapes)
        ):
            layer = event.source
            if self._is_roi_layer(layer):
                # Mark with metadata so it stays an ROI layer even if renamed
                self._mark_as_roi_layer(layer)
                renamed_to_roi = True

        current_text = self.layer_dropdown.currentText()
        roi_layer_names = list(self._get_roi_layers().keys())

        self.layer_dropdown.clear()
        if roi_layer_names:
            self.layer_dropdown.addItems(roi_layer_names)
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
            self._update_table_tooltip()
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
        self._mark_as_roi_layer(new_layer)
        self.layer_dropdown.setCurrentText(new_layer.name)

    def _link_layer_to_model(self, roi_layer: Shapes):
        """Link an ROIs layer to an ROIs table model."""
        # Disconnect previous model if it exists
        self._disconnect_table_model_signals()

        # Auto-assign names if the layer has shapes without names.
        self._auto_assign_roi_names(roi_layer)

        # Apply a consistent style to all shapes in the layer
        layer_color = self.color_manager.get_color_for_layer(roi_layer.name)
        roi_style = RoisStyle(color=layer_color)
        roi_style.color_all_shapes(roi_layer)

        # Create new model and link it to the table view
        self.roi_table_model = RoisTableModel(roi_layer, roi_style)
        self.roi_table_view.setModel(self.roi_table_model)

        # The model will listen to napari layer removal events
        self.viewer.layers.events.removed.connect(
            self.roi_table_model._on_layer_deleted
        )
        # Connect to layer name changes
        roi_layer.events.name.connect(self._on_layer_renamed)
        # Connect to model reset to update placeholder visibility
        self.roi_table_model.modelReset.connect(self._update_table_tooltip)

        # Update placeholder visibility
        self._update_table_tooltip()

    def _auto_assign_roi_names(self, roi_layer: Shapes) -> None:
        """Auto-assign names to ROIs if the layer has shapes without names.

        This handles cases where:
        - The "name" property doesn't exist
        - The "name" property is empty or shorter than the number of shapes
        - Some names are None or empty strings
        """
        if len(roi_layer.data) == 0:
            return

        # Get existing names, ensure proper length
        names = list(roi_layer.properties.get("name", []))
        n_shapes = len(roi_layer.data)
        while len(names) < n_shapes:  # pad with empty strings if needed
            names.append("")
        names = names[:n_shapes]      # trim if too long (defensive)

        # Check if any names are missing/invalid
        needs_update = any(
            not isinstance(name, str) or not name.strip() for name in names
        )
        if needs_update:
            # Let _update_roi_names logic take care of assigning names
            names = self._update_roi_names(names)

        roi_layer.properties = {"name": names}

    def _on_layer_renamed(self, event=None):
        """Handle layer renaming by updating the dropdown."""
        self._update_layer_dropdown()
        self.layer_dropdown.setCurrentText(event.source.name)

    def _disconnect_table_model_signals(self):
        """Disconnect signals from the ROIs table model."""
        if self.roi_table_model is not None:
            # Only disconnect layer events if the layer still exists
            if self.roi_table_model.layer is not None:
                self.roi_table_model.layer.events.data.disconnect(
                    self.roi_table_model._on_layer_data_changed
                )
                self.roi_table_model.layer.events.name.disconnect(
                    self._on_layer_renamed
                )
            # Always disconnect model/viewer events (don't depend on layer)
            self.roi_table_model.modelReset.disconnect(
                self._update_table_tooltip
            )
            self.viewer.layers.events.removed.disconnect(
                self.roi_table_model._on_layer_deleted
            )

    def _clear_roi_table_model(self):
        """Clear the ROIs table model."""
        self._disconnect_table_model_signals()
        self.roi_table_model = None
        self.roi_table_view.setModel(None)

    def _update_table_tooltip(self):
        """Update the table tooltip based on current state.

        Shows contextual hints:
        - How to create ROI layers when none exist
        - How to draw shapes when layer is empty
        - Usage tips when layer has shapes
        """
        layer_name = self.layer_dropdown.currentText()

        if not layer_name or layer_name == "Select a layer":
            # No ROI layers exist
            self.roi_table_view.setToolTip(
                "No ROI layers found.\n"
                "Click 'Add new layer' to create one."
            )
        elif (
            self.roi_table_model is None
            or self.roi_table_model.rowCount() == 0
        ):
            # Layer selected but no shapes
            self.roi_table_view.setToolTip(
                "No ROIs in this layer.\n"
                "Use the napari layer controls to draw shapes."
            )
        else:
            # Layer has shapes - show usage tips
            self.roi_table_view.setToolTip(
                "Click a row to select the shape.\n"
                "Press Delete to remove it.\n"
                "Double-click a name to rename."
            )

    def closeEvent(self, event):
        """Clean up signal connections when widget is closed."""
        # Disconnect all layer name signals
        for layer in list(self._connected_layers):
            self._disconnect_layer_name_signal(layer)

        # Disconnect viewer-level signals
        self.viewer.layers.events.inserted.disconnect(
            self._update_layer_dropdown
        )
        self.viewer.layers.events.removed.disconnect(
            self._on_layer_removed_from_viewer
        )

        # Clean up table model
        self._clear_roi_table_model()

        super().closeEvent(event)


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

    def __init__(
        self, shapes_layer: Shapes, roi_style: RoisStyle, parent=None
    ):
        """Initialize the ROIs table model with a Shapes layer and style.

        Parameters
        ----------
        shapes_layer : Shapes
            The napari Shapes layer containing the ROIs.
        roi_style : RoiStyle
            The style to apply to the ROIs.
        parent : QWidget, optional
            The parent widget.

        """
        super().__init__(parent)
        self.layer = shapes_layer
        self.roi_style = roi_style
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
        if self.layer is None:
            return

        if event.action in ["added", "removed"]:
            # Get current names, ensuring list length matches number of shapes
            current_names = list(self.layer.properties.get("name", []))
            n_shapes = len(self.layer.data)

            # Pad with empty strings if list is too short
            while len(current_names) < n_shapes:
                current_names.append("")

            # Update names for added shapes to ensure uniqueness
            updated_names = (
                self._update_roi_names(current_names)
                if event.action == "added"
                else current_names
            )
            self.layer.properties = {"name": updated_names}

            # Reapply the style to all shapes in the layer
            self.roi_style.color_all_shapes(self.layer)

            # Update the model
            self.beginResetModel()
            self.endResetModel()
        else:
            # Ensure currently edited shape maintains the correct style
            self.roi_style.color_current_shape(self.layer)

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

        Auto-assigns names only to shapes with empty/None names, or
        duplicate "ROI-<number>" pattern names. User-assigned names
        (anything that doesn't follow the ROI-<number> pattern) are
        always preserved, even if duplicated.

        Parameters
        ----------
        existing_names : list
            Current list of ROI names.

        Returns
        -------
        list
            Updated list with auto-assigned names where needed.

        """
        updated_names = existing_names.copy()

        # Find max number from existing ROI-<number> names
        auto_numbers = []
        for name in existing_names:
            if isinstance(name, str) and name.startswith("ROI-"):
                # Try parsing as ROI-<number>; ignore non-numeric suffixes
                # (e.g., "ROI-center" is a user name, not auto-assigned)
                with suppress(ValueError):
                    auto_numbers.append(int(name.split("-")[-1]))
        max_number = max(auto_numbers) if auto_numbers else 0

        # Track which ROI-<number> names we've seen (to detect duplicates)
        seen_roi_names = {}  # name -> first_index

        for i, name in enumerate(updated_names):
            needs_new_name = False

            if not isinstance(name, str) or not name.strip():
                # Empty/None → auto-assign
                needs_new_name = True
            elif name.startswith("ROI-"):
                # ROI-<number> pattern: check for duplicates
                if name in seen_roi_names:
                    needs_new_name = True  # Duplicate ROI-<number>
                else:
                    seen_roi_names[name] = i
            # else: user-assigned name like "center zone" → keep as-is

            if needs_new_name:
                max_number += 1
                new_name = f"ROI-{max_number}"
                updated_names[i] = new_name
                seen_roi_names[new_name] = i

        return updated_names
