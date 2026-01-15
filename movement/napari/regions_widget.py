"""Widget for defining regions of interest.

Regions are drawn as shapes in a napari Shapes layer and displayed in a table.
The widget can handle multiple region layers, allowing the user to select
which layer to work with via a dropdown. It also allows users to edit
the region names and applies consistent styling.

This module uses Qt's Model/View architecture to separate data from display:

- ``RegionsTableModel`` (Model): Wraps a napari Shapes layer and exposes
  region data (names, shape types) to the Qt framework. Listens to layer
  events and emits signals when data changes.
- ``RegionsTableView`` (View): Displays the model's data as a table. Handles
  user interactions like row selection and name editing.
- ``RegionsWidget``: Coordinates the model and view. Manages
  layer selection, creates/links models to views, and handles layer
  lifecycle events.

Data flow:
    napari Shapes layer <-> RegionsTableModel <-> RegionsTableView <-> User

See the `Qt Model/View framework
<https://doc.qt.io/qt-6/model-view-programming.html>`_
for more background.
"""

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

from movement.napari.layer_styles import RegionsColorManager, RegionsStyle

DEFAULT_REGION_NAME = "Un-named"


class RegionsWidget(QWidget):
    """Main widget for defining regions of interest.

    This widget provides a user interface for managing regions of interest
    drawn as shapes in napari. It coordinates the napari viewer,
    the RegionsTableModel, and the RegionsTableView.

    Features:

    - Dropdown to select existing region layers
    - Button to create new region layers
    - Table view displaying regions in the selected layer
    - Bidirectional selection sync: clicking a table row selects the
      shape in napari, and vice versa
    - Allows renaming regions via inline table editing
    - Consistent color styling per layer
    """

    def __init__(self, napari_viewer: Viewer, cmap_name="tab10", parent=None):
        """Initialise the regions widget.

        Parameters
        ----------
        napari_viewer : Viewer
            The napari viewer instance.
        cmap_name : str, optional
            Name of the napari colormap to use for region colors.
            Default is "tab10".
        parent : QWidget, optional
            The parent widget.

        """
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.color_manager = RegionsColorManager(cmap_name=cmap_name)
        self.region_table_model: RegionsTableModel | None = None
        self.region_table_view = RegionsTableView(self)
        self._connected_layers: set[Shapes] = set()

        self._setup_ui()
        self._connect_layer_signals()
        self._update_layer_dropdown()

    def _setup_ui(self):
        """Set up the user interface with two groupboxes.

        The first groupbox contains the region layer controls:
        a dropdown to select an existing Regions layer
        and a button to add a new Regions layer.
        The second groupbox contains the regions table view.
        """
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create layer controls group box
        layer_controls_group = QGroupBox("Layer to draw regions on")
        layer_controls_group.setToolTip(
            "Select an existing shapes layer to draw regions on, "
            "or add a new one.\nOnly shapes layers that start with "
            "'Region' are considered."
        )
        layer_controls_group.setLayout(self._setup_layer_controls_layout())
        main_layout.addWidget(layer_controls_group)

        # Create table view group box
        table_view_group = QGroupBox("Regions drawn in this layer")
        table_view_group.setLayout(self._setup_table_view_layout())
        main_layout.addWidget(table_view_group)

    def _setup_layer_controls_layout(self):
        """Create the region layer controls layout.

        Returns a QHBoxLayout containing:

        - Dropdown (QComboBox) for selecting region layers
        - "Add new layer" button (QPushButton)
        """
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
        """Create the table view layout.

        Returns a QVBoxLayout containing the RegionsTableView widget.
        """
        table_view_layout = QVBoxLayout()
        table_view_layout.addWidget(self.region_table_view)
        return table_view_layout

    def _connect_layer_signals(self):
        """Connect layer lifecycle signals to widget handlers.

        Handles layer insertion, removal, and name changes.
        """
        self.viewer.layers.events.inserted.connect(self._update_layer_dropdown)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

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

    def _is_region_layer(self, layer: Shapes) -> bool:
        """Check if a Shapes layer is a region layer.

        First checks for explicit metadata marker, then falls back to
        case-insensitive name matching.
        """
        if layer.metadata.get("movement_region_layer", False):
            return True
        return layer.name.upper().startswith("REGION")

    def _mark_as_region_layer(self, layer: Shapes) -> None:
        """Mark a Shapes layer as a region layer via metadata."""
        layer.metadata["movement_region_layer"] = True

    def _get_region_layers(self) -> dict[str, Shapes]:
        """Get all region layers.

        Returns a dictionary with layer names as keys and layers as values.
        """
        return {
            layer.name: layer
            for layer in self.viewer.layers
            if isinstance(layer, Shapes) and self._is_region_layer(layer)
        }

    def _on_layer_removed(self, event=None):
        """Handle layer removal from viewer.

        Disconnects name change signals from the removed layer and
        updates the dropdown to reflect available layers.
        """
        if event is not None and hasattr(event, "value"):
            layer = event.value
            if isinstance(layer, Shapes):
                self._disconnect_layer_name_signal(layer)
        self._update_layer_dropdown(event)

    def _update_layer_dropdown(self, event=None):
        """Refresh the layer dropdown with current region layers.

        Called when layers are added, removed, or renamed. Handles:

        - Connecting name change signals for new Shapes layers
        - Auto-marking layers renamed to "Region*" as region layers
        - Preserving the current selection when possible
        - Showing placeholder text when no region layers exist
        """
        # Connect to name change events for any new Shapes layers
        if event is not None and hasattr(event, "value"):
            layer = event.value
            if isinstance(layer, Shapes):
                self._connect_layer_name_signal(layer)

        # Check if a layer was renamed to a region name pattern
        # and mark it with metadata if so
        renamed_to_region = False
        if (
            event is not None
            and hasattr(event, "source")
            and isinstance(event.source, Shapes)
        ):
            layer = event.source
            if self._is_region_layer(layer):
                # Mark with metadata so it stays a region layer even if renamed
                self._mark_as_region_layer(layer)
                renamed_to_region = True

        current_text = self.layer_dropdown.currentText()
        region_layer_names = list(self._get_region_layers().keys())

        self.layer_dropdown.clear()
        if region_layer_names:
            self.layer_dropdown.addItems(region_layer_names)
            # Determine which layer to select
            if renamed_to_region:
                # Auto-select the newly renamed region layer
                self.layer_dropdown.setCurrentText(event.source.name)
            elif current_text in region_layer_names:
                # Next, try restoring the previous selection
                self.layer_dropdown.setCurrentText(current_text)
            else:
                # Fall back to the first layer
                self.layer_dropdown.setCurrentIndex(0)
        else:
            self.layer_dropdown.addItem("Select a layer")

    def _on_layer_selected(self, layer_name: str):
        """Handle layer selection from dropdown.

        - When a valid layer is selected, selects the layer in napari
          and links it to the table model for display.
        - When no layer is selected (placeholder text), clears the table model
          and the napari layer selection.
        """
        if not layer_name or layer_name == "Select a layer":
            self._clear_region_table_model()
            self.viewer.layers.selection.clear()
            self._update_table_tooltip()
            return

        region_layer = self._get_region_layers().get(layer_name)
        if region_layer is not None:
            # Select the layer in napari
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(region_layer)
            # Connect the region layer to the table model
            self._link_layer_to_model(region_layer)

    def _on_layer_renamed(self, event=None):
        """Handle layer renaming by updating the dropdown."""
        self._update_layer_dropdown()
        self.layer_dropdown.setCurrentText(event.source.name)

    def _add_new_layer(self):
        """Create a new Regions layer and select it."""
        new_layer = self.viewer.add_shapes(name="Regions")
        self._mark_as_region_layer(new_layer)
        self.layer_dropdown.setCurrentText(new_layer.name)

    def _link_layer_to_model(self, region_layer: Shapes):
        """Link a region layer to a new table model.

        This is the core method that connects the Model-View components:

        - Disconnects any previous model
        - Auto-assigns names to unnamed shapes
        - Applies consistent color styling
        - Creates a new RegionsTableModel for the layer
        - Connects model signals for data/selection sync
        """
        # Disconnect previous model if it exists
        self._disconnect_table_model_signals()

        # Auto-assign names if the layer has shapes without names.
        self._auto_assign_region_names(region_layer)

        # Apply a consistent style to all shapes in the layer
        layer_color = self.color_manager.get_color_for_layer(region_layer.name)
        region_style = RegionsStyle(color=layer_color)
        region_style.color_all_shapes(region_layer)

        # Create new model and link it to the table view
        self.region_table_model = RegionsTableModel(region_layer, region_style)
        self.region_table_view.setModel(self.region_table_model)

        # The model will listen to napari layer removal and renaming events
        self.viewer.layers.events.removed.connect(
            self.region_table_model._on_layer_deleted
        )
        # Connect to layer name changes
        region_layer.events.name.connect(self._on_layer_renamed)
        # Connect model reset signal to tooltip updater
        self.region_table_model.modelReset.connect(self._update_table_tooltip)

        # Update the tooltip based on the new model state
        self._update_table_tooltip()

    def _auto_assign_region_names(self, region_layer: Shapes) -> None:
        """Auto-assign names to regions if the layer has shapes without names.

        This handles cases where:

        - The "name" property doesn't exist
        - The "name" property is empty or shorter than the number of shapes
        - Some names are None or empty strings
        """
        if len(region_layer.data) == 0:
            return

        # Get existing names, ensure proper length
        names = list(region_layer.properties.get("name", []))
        n_shapes = len(region_layer.data)
        while len(names) < n_shapes:  # pad with empty strings if needed
            names.append("")

        # Check if any names are missing/invalid
        needs_update = any(
            not isinstance(name, str) or not name.strip() for name in names
        )
        if needs_update:
            names = _fill_empty_region_names(names)

        region_layer.properties = {"name": names}

    def _disconnect_table_model_signals(self):
        """Disconnect all signals from the current table model.

        Safely disconnects: layer data change events, layer set_data events,
        layer name change events, model reset signals, and viewer layer
        removal events.
        """
        if self.region_table_model is not None:
            # Only disconnect layer events if the layer still exists
            if self.region_table_model.layer is not None:
                self.region_table_model.layer.events.data.disconnect(
                    self.region_table_model._on_layer_data_changed
                )
                self.region_table_model.layer.events.set_data.disconnect(
                    self.region_table_model._on_layer_set_data
                )
                self.region_table_model.layer.events.name.disconnect(
                    self._on_layer_renamed
                )
            # Always disconnect model/viewer events (don't depend on layer)
            self.region_table_model.modelReset.disconnect(
                self._update_table_tooltip
            )
            self.viewer.layers.events.removed.disconnect(
                self.region_table_model._on_layer_deleted
            )

    def _clear_region_table_model(self):
        """Clear the current table model and disconnect from the view."""
        self._disconnect_table_model_signals()
        self.region_table_model = None
        self.region_table_view.setModel(None)

    def _update_table_tooltip(self):
        """Update the table tooltip based on current state.

        Shows contextual hints:

        - How to create region layers when none exist
        - How to draw shapes when layer is empty
        - Usage tips when layer has shapes
        """
        layer_name = self.layer_dropdown.currentText()

        if not layer_name or layer_name == "Select a layer":
            # No region layers exist
            self.region_table_view.setToolTip(
                "No region layers found.\nClick 'Add new layer' to create one."
            )
        elif (
            self.region_table_model is None
            or self.region_table_model.rowCount() == 0
        ):
            # Layer selected but no shapes
            self.region_table_view.setToolTip(
                "No regions in this layer.\n"
                "Use the napari layer controls to draw shapes."
            )
        else:
            # Layer has shapes - show usage tips
            self.region_table_view.setToolTip(
                "Click a row to select the shape.\n"
                "Press Delete to remove it.\n"
                "Double-click a name to rename."
            )

    def closeEvent(self, event):
        """Clean up signal connections when widget is closed.

        Overrides QWidget.closeEvent to ensure proper cleanup of:
            - Layer name change signals
            - Viewer-level layer insertion/removal signals
            - Table model connections
        """
        # Disconnect all layer name signals
        for layer in list(self._connected_layers):
            self._disconnect_layer_name_signal(layer)

        # Disconnect viewer-level signals
        self.viewer.layers.events.inserted.disconnect(
            self._update_layer_dropdown
        )
        self.viewer.layers.events.removed.disconnect(self._on_layer_removed)

        # Clean up table model
        self._clear_region_table_model()

        super().closeEvent(event)


class RegionsTableView(QTableView):
    """Table view for displaying drawn regions of interest.

    Displays region data from a RegionsTableModel in a two-column table
    (Name, Shape type). Handles user interactions:

    - Row selection syncs to shape selection in napari
    - Double-click on Name column enables inline editing
    """

    def __init__(self, parent=None):
        """Initialize the table view with selection and edit settings.

        Configures:

        - Row-based selection (clicking selects entire row)
        - Single selection mode (one row at a time)
        - Double-click or key press to edit Name column
        """
        super().__init__(parent=parent)
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)
        self.setEditTriggers(
            QTableView.DoubleClicked | QTableView.EditKeyPressed
        )
        self.current_model: RegionsTableModel | None = None

    def setModel(self, model):
        """Set the table model and connect selection signals.

        Overrides QTableView.setModel to additionally connect the
        selection changed signal for syncing with napari layer selection.
        """
        super().setModel(model)
        self.current_model = model

        if model is not None:
            self.selectionModel().selectionChanged.connect(
                self._on_selection_changed
            )

    def _on_selection_changed(self, selected, deselected):
        """Sync table row selection to napari shape selection.

        When user clicks a row in the table, selects the corresponding
        shape in the napari Shapes layer.
        """
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


class RegionsTableModel(QAbstractTableModel):
    """Table model exposing region data from a Shapes layer.

    Wraps a napari Shapes layer and provides data to RegionsTableView:

    - Column 0: Region name (from layer.properties["name"])
    - Column 1: Shape type (e.g., "rectangle", "polygon")

    Listens to layer data events and emits Qt signals when shapes are
    added, removed, or modified. Also handles auto-naming of new shapes
    and applies consistent styling via RegionsStyle.
    """

    def __init__(
        self, shapes_layer: Shapes, region_style: RegionsStyle, parent=None
    ):
        """Initialize the model with a Shapes layer and style.

        Parameters
        ----------
        shapes_layer : Shapes
            The napari Shapes layer containing the regions.
        region_style : RegionsStyle
            The style to apply to the regions.
        parent : QWidget, optional
            The parent widget.

        """
        super().__init__(parent)
        self.layer = shapes_layer
        self.region_style = region_style
        # Track shape count to detect new shapes
        self._last_shape_count = len(shapes_layer.data)
        # Listen to layer data changes (drawing, editing, deleting shapes)
        self.layer.events.data.connect(self._on_layer_data_changed)
        # Listen to set_data events (copy-paste emits this, not data)
        self.layer.events.set_data.connect(self._on_layer_set_data)

    def rowCount(self, parent=QModelIndex()):  # noqa: B008
        """Return the number of regions (shapes) in the layer."""
        return len(self.layer.data) if self.layer else 0

    def columnCount(self, parent=QModelIndex()):  # noqa: B008
        """Return 2 columns: Name and Shape type."""
        return 2 if self.layer else 0

    def data(self, index, role=Qt.DisplayRole):
        """Return cell data for display or editing."""
        if not index.isValid():
            return None

        row, col = index.row(), index.column()

        if row >= len(self.layer.data):
            return None

        if role == Qt.DisplayRole:
            if col == 0:
                return self._get_region_name_for_row(row)
            elif col == 1:
                return (
                    self.layer.shape_type[row]
                    if row < len(self.layer.shape_type)
                    else ""
                )
        elif role == Qt.EditRole and col == 0:
            # Return editable data for the Name column
            return self._get_region_name_for_row(row)
        return None

    def flags(self, index):
        """Return item flags (editable for Name column only)."""
        if not index.isValid():
            return Qt.NoItemFlags

        if index.column() == 0:  # Make only the Name column editable
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def setData(self, index, value, role=Qt.EditRole):
        """Update region name when user edits the Name column.

        Updates the layer.properties["name"] list and emits dataChanged.
        Only the Name column (column 0) is editable.
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

            names[row] = str(value)
            self.layer.properties = {"name": names}
            self.dataChanged.emit(index, index)
            return True

        return False

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Return header labels: 'Name' and 'Shape type' for columns."""
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return ["Name", "Shape type"][section]
        else:  # Vertical orientation
            return str(section)  # Return the row index as a string

    def _get_region_name_for_row(self, row):
        """Get the region name for a given row index from layer properties."""
        names = self.layer.properties.get("name", [])
        return names[row] if row < len(names) else ""

    def _on_layer_data_changed(self, event=None):
        """Handle data events from drawing, editing, or deleting shapes.

        For "added" events (drawing new shapes): assigns default name.
        For "removed" events: syncs model with remaining shapes.
        For other events (e.g., moving/resizing): updates current shape style.
        """
        if self.layer is None:
            return

        n_shapes = len(self.layer.data)

        if event.action == "added":
            # New shape drawn - assign default name to override napari's
            # property copying behavior
            self._sync_names_on_shape_change(
                n_shapes, assign_default_to_new=True
            )
        elif event.action == "removed":
            # Shape deleted - just sync names list
            self._sync_names_on_shape_change(n_shapes)
        else:
            # Shape edited (moved, resized) - just update styling
            self.region_style.color_current_shape(self.layer)

    def _on_layer_set_data(self, event=None):
        """Handle set_data events from copy-paste operations.

        Copy-paste in napari emits set_data (not data) events.
        We preserve the copied name (as is napari's default behavior).
        """
        if self.layer is None:
            return

        n_shapes = len(self.layer.data)
        if n_shapes != self._last_shape_count:
            # Shape count changed via copy-paste - sync without overriding name
            self._sync_names_on_shape_change(n_shapes)

    def _sync_names_on_shape_change(
        self, n_shapes: int, assign_default_to_new: bool = False
    ):
        """Sync names list with current shape count and update model.

        Parameters
        ----------
        n_shapes : int
            Current number of shapes in the layer.
        assign_default_to_new : bool, optional
            If True, assigns DEFAULT_REGION_NAME to newly added shapes.
            Use for drawn shapes (not copy-pasted ones). Default is False.

        """
        current_names = list(self.layer.properties.get("name", []))

        # Pad if list is too short
        while len(current_names) < n_shapes:
            current_names.append(DEFAULT_REGION_NAME)

        # Truncate if list is too long (shapes were removed)
        current_names = current_names[:n_shapes]

        # Override names for newly drawn shapes
        if assign_default_to_new and n_shapes > self._last_shape_count:
            for i in range(self._last_shape_count, n_shapes):
                current_names[i] = DEFAULT_REGION_NAME

        self.layer.properties = {"name": current_names}
        self._last_shape_count = n_shapes

        # Reapply styling and update model
        self.region_style.color_all_shapes(self.layer)
        self.beginResetModel()
        self.endResetModel()

    def _on_layer_deleted(self, event=None):
        """Handle deletion of the associated Shapes layer from viewer.

        Disconnects from layer events, clears the layer reference,
        and resets the model to empty state.
        """
        # Only reset the model if the layer being removed
        # is the one we are currently using.
        if event.value == self.layer:
            self.layer.events.data.disconnect(self._on_layer_data_changed)
            self.layer.events.set_data.disconnect(self._on_layer_set_data)
            self.layer = None
            self.beginResetModel()
            self.endResetModel()


def _fill_empty_region_names(existing_names: list) -> list:
    """Fill empty/None region names with DEFAULT_REGION_NAME.

    Parameters
    ----------
    existing_names : list
        Current list of region names.

    Returns
    -------
    list
        Updated list with default names where needed.

    """
    return [
        name if isinstance(name, str) and name.strip() else DEFAULT_REGION_NAME
        for name in existing_names
    ]
