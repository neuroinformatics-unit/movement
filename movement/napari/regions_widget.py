"""Widget for defining regions of interest.

This module uses Qt's Model/View architecture to separate data from display.
See our `napari plugin development guide
<https://movement.neuroinformatics.dev/dev/community/contributing.html#developing-the-napari-plugin>`_
for more background.
"""

import re

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

from movement.napari.layer_styles import RegionsStyle, _sample_colormap

DEFAULT_REGION_NAME = "region"
DROPDOWN_PLACEHOLDER = "No region layers"

# Metadata keys stored on napari Shapes layers managed by this widget.
# - REGIONS_LAYER_KEY: marks the layer as a movement regions layer on creation.
# - REGIONS_COLOR_IDX_KEY stores the palette index assigned to the regions
#   layer so its color remains stable across re-linking.
REGIONS_LAYER_KEY: str = "movement_regions_layer"
REGIONS_COLOR_IDX_KEY: str = "movement_regions_color_idx"

# Fixed palette of colours assigned sequentially
# to regions layers as they are first linked to the widget.
REGIONS_COLORS: list[tuple] = _sample_colormap(10, "tab10")


class RegionsWidget(QWidget):
    """Main widget for defining regions of interest.

    This widget provides a user interface for managing regions of interest
    drawn as shapes in napari. It coordinates the napari viewer,
    the RegionsTableModel, and the RegionsTableView.

    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialise the regions widget.

        Parameters
        ----------
        napari_viewer
            The napari viewer instance.
        parent
            The parent widget.

        """
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self._next_color_idx = 0
        self.region_table_model: RegionsTableModel | None = None
        self.region_table_view = RegionsTableView(self)

        # Guard flag to prevent circular updates during layer selection syncing
        # between napari's layer list the dropdown in this widget.
        self._syncing_layer_selection = False

        self._setup_regions_ui()
        self._connect_layer_signals()
        self._update_layer_dropdown()

    def _setup_regions_ui(self):
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
        layer_controls_group.setLayout(self._setup_region_layer_controls())
        main_layout.addWidget(layer_controls_group)

        # Create table view group box
        table_view_group = QGroupBox("Regions drawn in this layer")
        table_view_group.setLayout(self._setup_regions_table())
        main_layout.addWidget(table_view_group)

    def _setup_region_layer_controls(self):
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

    def _setup_regions_table(self):
        """Create the table view layout.

        Returns a QVBoxLayout containing the RegionsTableView widget.
        """
        table_view_layout = QVBoxLayout()
        table_view_layout.addWidget(self.region_table_view)
        return table_view_layout

    def _connect_layer_signals(self):
        """Connect layer lifecycle signals to widget handlers.

        Handles layer insertion, removal, and selection changes.
        """
        self.viewer.layers.events.inserted.connect(self._update_layer_dropdown)
        self.viewer.layers.events.removed.connect(self._update_layer_dropdown)
        self.viewer.layers.selection.events.changed.connect(
            self._on_napari_layer_selection_changed
        )

    def _is_region_layer(self, layer) -> bool:
        """Check if a layer is a movement regions layer."""
        return isinstance(layer, Shapes) and bool(
            layer.metadata.get(REGIONS_LAYER_KEY, False)
        )

    def _get_region_layers(self) -> dict[str, Shapes]:
        """Get all region layers.

        Returns a dictionary with layer names as keys and layers as values.
        """
        return {
            layer.name: layer
            for layer in self.viewer.layers
            if self._is_region_layer(layer)
        }

    def _update_layer_dropdown(self, _event=None):
        """Refresh the layer dropdown with current region layers.

        Called when layers are added, removed, or renamed. Preserves the
        current selection when possible; falls back to the first layer.
        Shows placeholder text when no region layers exist.
        """
        current_text = self.layer_dropdown.currentText()
        region_layer_names = list(self._get_region_layers().keys())

        self.layer_dropdown.clear()
        if region_layer_names:
            self.layer_dropdown.setStyleSheet("")
            self.layer_dropdown.addItems(region_layer_names)
            if current_text in region_layer_names:
                self.layer_dropdown.setCurrentText(current_text)
            else:
                self.layer_dropdown.setCurrentIndex(0)
        else:
            self.layer_dropdown.setStyleSheet("color: gray;")
            self.layer_dropdown.addItem(DROPDOWN_PLACEHOLDER)
            self.layer_dropdown.model().item(0).setEnabled(False)

    def _on_layer_selected(self, layer_name: str):
        """Handle layer selection from dropdown.

        - When a valid layer is selected, selects the layer in napari
          and links it to the table model for display.
        - When no layer is selected (placeholder text), clears the table model
          and the napari layer selection.
        """
        if not layer_name or layer_name == DROPDOWN_PLACEHOLDER:
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

    def _on_napari_layer_selection_changed(self, event=None):
        """Sync napari layer list selection to the dropdown and table.

        When the user clicks a region layer in napari's layer list,
        the dropdown and table update to reflect that layer.
        """
        # Return early if we're already syncing to avoid circular updates
        if self._syncing_layer_selection:
            return

        active = self.viewer.layers.selection.active
        # Return early if the active layer is not a region layer
        if not self._is_region_layer(active):
            return

        # Return early if the active layer is already selected in the dropdown
        if self.layer_dropdown.currentText() == active.name:
            return

        # Sync the dropdown to match the active layer (in a guarded block)
        self._syncing_layer_selection = True
        self.layer_dropdown.setCurrentText(active.name)
        self._syncing_layer_selection = False

    def _on_layer_renamed(self, event=None):
        """Handle layer renaming by updating the dropdown."""
        self._update_layer_dropdown()
        self.layer_dropdown.setCurrentText(event.source.name)

    def _add_new_layer(self):
        """Create a new Regions layer and select it."""
        new_layer = self.viewer.add_shapes(
            name="regions",
            metadata={REGIONS_LAYER_KEY: True},
        )
        self.layer_dropdown.setCurrentText(new_layer.name)

    def _link_layer_to_model(self, region_layer: Shapes):
        """Link a regions layer to a new table model.

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

        # On first link, assign the next palette color and apply it to all
        # existing shapes (also primes current_* so the first drawn shape
        # gets the palette color). On re-link, napari restores each layer's
        # own current_* automatically — no intervention needed.
        if REGIONS_COLOR_IDX_KEY not in region_layer.metadata:
            region_layer.metadata[REGIONS_COLOR_IDX_KEY] = self._next_color_idx
            idx = self._next_color_idx % len(REGIONS_COLORS)
            self._next_color_idx += 1
            RegionsStyle(color=REGIONS_COLORS[idx]).set_color_all_shapes(
                region_layer
            )

        # Create new model and link it to the table view
        self.region_table_model = RegionsTableModel(region_layer)
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
                self.region_table_model.layer.events.edge_color.disconnect(
                    self.region_table_model._on_edge_color_changed
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

        if not layer_name or layer_name == DROPDOWN_PLACEHOLDER:
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
            - Viewer-level layer insertion/removal/selection signals
            - Table model connections
        """
        # Disconnect viewer-level signals
        self.viewer.layers.events.inserted.disconnect(
            self._update_layer_dropdown
        )
        self.viewer.layers.events.removed.disconnect(
            self._update_layer_dropdown
        )
        self.viewer.layers.selection.events.changed.disconnect(
            self._on_napari_layer_selection_changed
        )

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

        # Guard flag to prevent circular updates between the two sync
        # handlers: row selection → napari shape selection → row selection...
        self._syncing_row_selection = False

    def setModel(self, model):
        """Set the table model and connect selection signals.

        Overrides QTableView.setModel to additionally connect the
        selection changed signal for syncing with napari layer selection.
        """
        # Disconnect the (view-managed) highlight event from the previous layer
        prev_layer = getattr(self.current_model, "layer", None)
        if self.current_model is not None and prev_layer is not None:
            prev_layer.events.highlight.disconnect(
                self._on_shape_selection_changed
            )

        super().setModel(model)
        self.current_model = model

        if model is not None:
            self.selectionModel().selectionChanged.connect(
                self._on_row_selection_changed
            )
            if model.layer is not None:
                model.layer.events.highlight.connect(
                    self._on_shape_selection_changed
                )

    def _on_row_selection_changed(self, selected, deselected):
        """Sync row selection in the table to shape selection in napari."""
        if self._syncing_row_selection:
            return
        if self.current_model is None or self.current_model.layer is None:
            return

        indexes = selected.indexes()
        if not indexes:
            return

        row = indexes[0].row()
        if row < len(self.current_model.layer.data):
            self._syncing_row_selection = True
            self.current_model.layer.selected_data = {row}
            self._syncing_row_selection = False

    def _on_shape_selection_changed(self, event=None):
        """Sync shape selection in napari to row highlight in the table."""
        if self._syncing_row_selection:
            return
        if self.current_model is None or self.current_model.layer is None:
            return

        selected = self.current_model.layer.selected_data
        self._syncing_row_selection = True
        if len(selected) == 1:
            self.selectRow(next(iter(selected)))
        else:
            self.clearSelection()
        self._syncing_row_selection = False


class RegionsTableModel(QAbstractTableModel):
    """Table model exposing region data from a Shapes layer.

    Wraps a napari Shapes layer and provides data to RegionsTableView:

    - Column 0: Region name (from layer.properties["name"])
    - Column 1: Shape type (e.g., "rectangle", "polygon")

    Listens to layer data events and emits Qt signals when shapes are
    added, removed, or modified. Also handles auto-naming of new shapes.
    """

    def __init__(self, shapes_layer: Shapes, parent=None):
        """Initialize the model with a Shapes layer.

        Parameters
        ----------
        shapes_layer
            The napari Shapes layer containing the regions.
        parent
            The parent widget.

        """
        super().__init__(parent)
        self.layer = shapes_layer
        # Track shape count to detect new shapes
        self._last_shape_count = len(shapes_layer.data)
        # Guard flag: True between "adding" and "added" data events.
        # Prevents the interleaved set_data event from processing drawn shapes
        self._adding_shape = False
        # Listen to layer data changes (drawing, editing, deleting shapes)
        self.layer.events.data.connect(self._on_layer_data_changed)
        # Listen to set_data events (copy-paste emits this, not data)
        self.layer.events.set_data.connect(self._on_layer_set_data)
        # Keep text colour tethered to edge colour
        self.layer.events.edge_color.connect(self._on_edge_color_changed)

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

            while len(names) <= row:  # pragma: no cover
                names.append("")

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

        - For "added" events (drawing new shapes): assigns default name.
        - For "removed" events: syncs model with remaining shapes.
        - The "adding" event sets a guard flag so that the interleaved
          set_data event (fired between "adding" and "added") defers
          naming to this handler.
        - Moves/resizes do not affect names and are ignored.
        """
        if self.layer is None:
            return

        n_shapes = len(self.layer.data)

        if event.action == "adding":
            self._adding_shape = True
        elif event.action == "added":
            self._adding_shape = False
            self._sync_names_on_shape_change(
                n_shapes, use_default_name=True
            )
        elif event.action == "removed":
            self._sync_names_on_shape_change(n_shapes)

    def _on_edge_color_changed(self, event=None):
        """Tether text colour to edge colour when the user recolours shapes."""
        if self.layer is not None and len(self.layer.data) > 0:
            self.layer.text.color = self.layer.edge_color

    def _on_layer_set_data(self, event=None):
        """Handle set_data events from copy-paste operations.

        Copy-paste in napari emits set_data (not data) events. Newly pasted
        shapes get unique names derived from the copied name (e.g. pasting
        "burrow" when "burrow" already exists yields "burrow [1]").

        When drawing, napari also fires set_data between the "adding" and
        "added" data events. In that case we skip here and let
        _on_layer_data_changed handle naming with the correct default.
        """
        if self.layer is None:
            return
        if self._adding_shape:
            return

        n_shapes = len(self.layer.data)
        if n_shapes != self._last_shape_count:
            self._sync_names_on_shape_change(n_shapes)

    def _sync_names_on_shape_change(
        self,
        n_shapes: int,
        use_default_name: bool = False,
    ):
        """Sync names list with current shape count and update model.

        Parameters
        ----------
        n_shapes
            Current number of shapes in the layer.
        use_default_name
            If True, newly added shapes get a unique default name
            (e.g. "region", "region [1]"). Use for drawn shapes.
            If False, the existing name (e.g. from a copy-paste) is
            kept as the base and made unique if needed.
            Default is False.

        """
        current_names = list(self.layer.properties.get("name", []))

        # Pad if list is too short (probably overly defensive)
        while len(current_names) < n_shapes:  # pragma: no cover
            current_names.append(DEFAULT_REGION_NAME)

        # Truncate if list is too long (shapes were removed)
        current_names = current_names[:n_shapes]

        # Assign unique names to newly added shapes.
        # For drawn shapes (use_default_name=True), we override
        # whatever napari propagated from the selected shape.
        # For pasted shapes (use_default_name=False), we keep the
        # copied name as the base and just make it unique.
        if n_shapes > self._last_shape_count:
            for i in range(self._last_shape_count, n_shapes):
                base = current_names[i]
                if use_default_name or not isinstance(base, str):
                    base = DEFAULT_REGION_NAME
                current_names[i] = _unique_name(base, current_names[:i])

        # text.string must be set before properties.
        if n_shapes > 0:
            self.layer.text.string = "{name}"

        self.layer.properties = {"name": current_names}
        self._last_shape_count = n_shapes

        # text.refresh() (inside the properties setter) resets text colours
        # to the default encoding, so re-tether them to edge colours.
        if n_shapes > 0:
            self.layer.text.color = self.layer.edge_color

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
            self.layer.events.edge_color.disconnect(
                self._on_edge_color_changed
            )
            self.layer = None
            self.beginResetModel()
            self.endResetModel()


def _unique_name(base: str, existing_names: list) -> str:
    """Return base if not already taken, else base [1], base [2], etc.

    Parameters
    ----------
    base
        Desired name.
    existing_names
        Names already in use.

    Returns
    -------
    str
        A name that does not appear in existing_names.

    """
    # Strip existing " [N]" suffixes
    root = re.sub(r" \[\d+\]$", "", base)
    if root not in existing_names:
        return root
    i = 1
    while f"{root} [{i}]" in existing_names:
        i += 1
    return f"{root} [{i}]"


def _fill_empty_region_names(existing_names: list) -> list:
    """Fill empty/None region names with DEFAULT_REGION_NAME.

    Parameters
    ----------
    existing_names
        Current list of region names.

    Returns
    -------
    list
        Updated list with default names where needed.

    """
    result = list(existing_names)
    for i, name in enumerate(result):
        if not isinstance(name, str) or not name.strip():
            result[i] = _unique_name(DEFAULT_REGION_NAME, result[:i])
    return result
