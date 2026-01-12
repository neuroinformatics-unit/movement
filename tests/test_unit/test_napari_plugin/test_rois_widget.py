"""Unit tests for the ROIs widget in the napari plugin."""

from contextlib import nullcontext as does_not_raise

import pytest
from napari.layers import Shapes
from qtpy.QtCore import QModelIndex, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QPushButton,
    QTableView,
)

from movement.napari.rois_widget import (
    RoisTableView,
    RoisWidget,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Previous color_by key.*:UserWarning"
)


# ------------------- Fixtures for ROIs widget tests -------------------------#
@pytest.fixture
def sample_shapes_data():
    """Return sample shapes data for testing."""
    return [
        [[0, 0], [0, 10], [10, 10], [10, 0]],
        [[20, 20], [20, 30], [30, 30], [30, 20]],
        [[40, 40], [40, 50], [50, 50], [50, 40]],
    ]


@pytest.fixture
def rois_widget(make_napari_viewer_proxy):
    """Return a RoisWidget with an empty viewer."""
    viewer = make_napari_viewer_proxy()
    return RoisWidget(viewer)


@pytest.fixture
def rois_widget_with_layer(make_napari_viewer_proxy, sample_shapes_data):
    """Return a RoisWidget with a viewer and shapes layer containing ROIs."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(
        sample_shapes_data[:2],
        shape_type="polygon",
        name="ROIs",
    )
    layer.properties = {"name": ["ROI-1", "ROI-2"]}
    widget = RoisWidget(viewer)
    return widget, layer


# ------------------- Tests for widget instantiation -------------------------#
class TestRoisWidgetInstantiation:
    """Tests for RoisWidget instantiation and UI setup."""

    def test_rois_widget_instantiation(self, make_napari_viewer_proxy):
        """Test that the ROIs widget is properly instantiated."""
        viewer = make_napari_viewer_proxy()
        widget = RoisWidget(viewer)

        assert widget.viewer == viewer
        assert widget.roi_table_model is None  # No model until layer selected
        assert isinstance(widget.roi_table_view, RoisTableView)
        assert len(widget._connected_layers) == 0

    def test_rois_widget_has_expected_ui_elements(self, rois_widget):
        """Test that the ROIs widget has all expected UI elements."""
        group_boxes = rois_widget.findChildren(QGroupBox)
        assert len(group_boxes) == 2

        assert rois_widget.findChild(QComboBox) is not None
        assert rois_widget.findChild(
            QPushButton, "Add new layer"
        ) is not None or (rois_widget.add_layer_button is not None)
        assert rois_widget.findChild(QTableView) is not None

    def test_rois_widget_dropdown_shows_placeholder_when_no_layers(
        self, rois_widget
    ):
        """Test dropdown shows placeholder when no ROI layers exist."""
        assert rois_widget.layer_dropdown.currentText() == "Select a layer"

    def test_rois_widget_with_custom_colormap(self, make_napari_viewer_proxy):
        """Test that the widget can be instantiated with a custom colormap."""
        viewer = make_napari_viewer_proxy()
        widget = RoisWidget(viewer, cmap_name="viridis")

        assert widget.color_manager.cmap_name == "viridis"


# ------------------- Tests for layer dropdown functionality -----------------#
class TestLayerDropdown:
    """Tests for layer dropdown population and selection."""

    def test_dropdown_populated_with_existing_roi_layer(
        self, make_napari_viewer_proxy
    ):
        """Test dropdown is populated when ROI layer exists at init."""
        viewer = make_napari_viewer_proxy()
        viewer.add_shapes(name="ROIs")
        widget = RoisWidget(viewer)

        assert widget.layer_dropdown.count() == 1
        assert widget.layer_dropdown.currentText() == "ROIs"

    def test_dropdown_updated_on_layer_added(self, rois_widget):
        """Test dropdown is updated when a new ROI layer is added."""
        assert rois_widget.layer_dropdown.currentText() == "Select a layer"

        rois_widget.viewer.add_shapes(name="ROIs")

        assert rois_widget.layer_dropdown.count() == 1
        assert rois_widget.layer_dropdown.currentText() == "ROIs"

    def test_dropdown_updated_on_layer_removed(self, rois_widget_with_layer):
        """Test dropdown is updated when an ROI layer is removed."""
        widget, layer = rois_widget_with_layer

        assert widget.layer_dropdown.count() == 1

        widget.viewer.layers.remove(layer)

        assert widget.layer_dropdown.currentText() == "Select a layer"

    def test_dropdown_ignores_non_roi_layers(self, make_napari_viewer_proxy):
        """Test dropdown ignores non-ROI shapes layers."""
        viewer = make_napari_viewer_proxy()
        viewer.add_shapes(name="Other shapes")
        widget = RoisWidget(viewer)

        assert widget.layer_dropdown.currentText() == "Select a layer"

    def test_dropdown_includes_layer_with_roi_metadata(
        self, make_napari_viewer_proxy
    ):
        """Test dropdown includes layers marked with ROI metadata."""
        viewer = make_napari_viewer_proxy()
        layer = viewer.add_shapes(name="Custom name")
        layer.metadata["movement_roi_layer"] = True
        widget = RoisWidget(viewer)

        assert widget.layer_dropdown.count() == 1
        assert widget.layer_dropdown.currentText() == "Custom name"

    def test_dropdown_preserves_selection_on_update(
        self, make_napari_viewer_proxy
    ):
        """Test dropdown preserves current selection when updated."""
        viewer = make_napari_viewer_proxy()
        viewer.add_shapes(name="ROIs")
        viewer.add_shapes(name="ROIs [1]")
        widget = RoisWidget(viewer)
        widget.layer_dropdown.setCurrentText("ROIs [1]")

        viewer.add_shapes(name="ROIs [2]")

        assert widget.layer_dropdown.currentText() == "ROIs [1]"


# ------------------- Tests for layer selection ------------------------------#
class TestLayerSelection:
    """Tests for layer selection behavior."""

    def test_selecting_layer_from_dropdown_selects_in_viewer(
        self, make_napari_viewer_proxy
    ):
        """Test that selecting a layer from dropdown selects it in viewer."""
        viewer = make_napari_viewer_proxy()
        layer = viewer.add_shapes(name="ROIs")
        RoisWidget(viewer)

        assert layer in viewer.layers.selection

    def test_selecting_layer_links_to_model(self, rois_widget_with_layer):
        """Test that selecting a layer creates a linked table model."""
        widget, layer = rois_widget_with_layer

        assert widget.roi_table_model is not None
        assert widget.roi_table_model.layer == layer

    def test_selecting_placeholder_clears_model(self, rois_widget_with_layer):
        """Test that selecting placeholder clears the table model."""
        widget, layer = rois_widget_with_layer

        assert widget.roi_table_model is not None

        widget.viewer.layers.remove(layer)

        assert widget.roi_table_model is None


# ------------------- Tests for add new layer button -------------------------#
class TestAddNewLayer:
    """Tests for the Add new layer button."""

    def test_add_new_layer_creates_roi_layer(self, rois_widget):
        """Test that clicking Add new layer creates a new ROI layer."""
        rois_widget._add_new_layer()

        assert len(rois_widget.viewer.layers) == 1
        assert isinstance(rois_widget.viewer.layers[0], Shapes)
        assert rois_widget.viewer.layers[0].name.startswith("ROIs")

    def test_add_new_layer_marks_with_metadata(self, rois_widget):
        """Test that new layer is marked with ROI metadata."""
        rois_widget._add_new_layer()

        assert (
            rois_widget.viewer.layers[0].metadata.get("movement_roi_layer")
            is True
        )

    def test_add_new_layer_selects_in_dropdown(self, rois_widget):
        """Test that new layer is selected in dropdown."""
        rois_widget._add_new_layer()

        assert (
            rois_widget.layer_dropdown.currentText()
            == rois_widget.viewer.layers[0].name
        )

    def test_add_multiple_layers_increments_name(self, rois_widget):
        """Test that multiple new layers get incremented names."""
        rois_widget._add_new_layer()
        rois_widget._add_new_layer()

        layer_names = [layer.name for layer in rois_widget.viewer.layers]
        assert len(set(layer_names)) == 2  # Names should be unique


# ------------------- Tests for RoisTableModel -------------------------------#
class TestRoisTableModel:
    """Tests for RoisTableModel functionality."""

    def test_model_row_count_matches_shapes(self, rois_widget_with_layer):
        """Test that model rowCount matches number of shapes."""
        widget, _layer = rois_widget_with_layer

        assert widget.roi_table_model.rowCount() == 2

    def test_model_column_count(self, rois_widget_with_layer):
        """Test that model has 2 columns (Name and Shape type)."""
        widget, _layer = rois_widget_with_layer

        assert widget.roi_table_model.columnCount() == 2

    def test_model_header_labels(self, rois_widget_with_layer):
        """Test that model header labels are correct."""
        widget, _layer = rois_widget_with_layer

        assert widget.roi_table_model.headerData(0, Qt.Horizontal) == "Name"
        assert (
            widget.roi_table_model.headerData(1, Qt.Horizontal) == "Shape type"
        )

    def test_model_data_returns_roi_name(self, rois_widget_with_layer):
        """Test that model returns ROI name for column 0."""
        widget, _layer = rois_widget_with_layer

        index = widget.roi_table_model.index(0, 0)
        assert widget.roi_table_model.data(index, Qt.DisplayRole) == "ROI-1"

    def test_model_data_returns_shape_type(self, rois_widget_with_layer):
        """Test that model returns shape type for column 1."""
        widget, _layer = rois_widget_with_layer

        index = widget.roi_table_model.index(0, 1)
        assert widget.roi_table_model.data(index, Qt.DisplayRole) == "polygon"

    def test_model_setData_updates_roi_name(self, rois_widget_with_layer):
        """Test that setData updates the ROI name."""
        widget, layer = rois_widget_with_layer

        index = widget.roi_table_model.index(0, 0)
        result = widget.roi_table_model.setData(index, "New Name", Qt.EditRole)

        assert result is True
        assert layer.properties["name"][0] == "New Name"

    def test_model_setData_does_not_update_shape_type(
        self, rois_widget_with_layer
    ):
        """Test that setData returns False for shape type column."""
        widget, _layer = rois_widget_with_layer

        index = widget.roi_table_model.index(0, 1)
        result = widget.roi_table_model.setData(
            index, "rectangle", Qt.EditRole
        )

        assert result is False

    def test_model_flags_name_column_editable(self, rois_widget_with_layer):
        """Test that name column is editable."""
        widget, _layer = rois_widget_with_layer

        index = widget.roi_table_model.index(0, 0)
        flags = widget.roi_table_model.flags(index)

        assert flags & Qt.ItemIsEditable

    def test_model_flags_shape_type_not_editable(self, rois_widget_with_layer):
        """Test that shape type column is not editable."""
        widget, _layer = rois_widget_with_layer

        index = widget.roi_table_model.index(0, 1)
        flags = widget.roi_table_model.flags(index)

        assert not (flags & Qt.ItemIsEditable)


# ------------------- Tests for shape events ---------------------------------#
class TestShapeEvents:
    """Tests for handling shape add/remove events."""

    def test_adding_shape_updates_model(self, rois_widget_with_layer):
        """Test that adding a shape updates the model."""
        widget, layer = rois_widget_with_layer

        initial_count = widget.roi_table_model.rowCount()
        layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])

        assert widget.roi_table_model.rowCount() == initial_count + 1

    def test_removing_shape_updates_model(self, rois_widget_with_layer):
        """Test that removing a shape updates the model."""
        widget, layer = rois_widget_with_layer

        initial_count = widget.roi_table_model.rowCount()
        layer.selected_data = {0}
        layer.remove_selected()

        assert widget.roi_table_model.rowCount() == initial_count - 1

    def test_new_shape_gets_auto_name(self, rois_widget_with_layer):
        """Test that new shapes get auto-assigned names."""
        widget, layer = rois_widget_with_layer

        layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])

        names = list(layer.properties.get("name", []))
        assert len(names) == 3
        assert names[2] == "ROI-3"
        del widget  # Prevent GC warning


# ------------------- Tests for ROI auto-naming ------------------------------#
class TestROIAutoNaming:
    """Tests for ROI auto-naming functionality."""

    def test_auto_naming_fills_empty_names(self, make_napari_viewer_proxy):
        """Test that empty names are filled with auto-generated names."""
        viewer = make_napari_viewer_proxy()
        layer = viewer.add_shapes(
            [[[0, 0], [0, 10], [10, 10], [10, 0]]],
            shape_type="polygon",
            name="ROIs",
        )
        layer.properties = {"name": [""]}

        RoisWidget(viewer)

        names = list(layer.properties.get("name", []))
        assert names[0] == "ROI-1"

    def test_auto_naming_handles_none_values(self, make_napari_viewer_proxy):
        """Test that None values in names are replaced."""
        viewer = make_napari_viewer_proxy()
        layer = viewer.add_shapes(
            [[[0, 0], [0, 10], [10, 10], [10, 0]]],
            shape_type="polygon",
            name="ROIs",
        )
        layer.properties = {"name": [None]}

        RoisWidget(viewer)

        names = list(layer.properties.get("name", []))
        assert names[0] == "ROI-1"

    def test_auto_naming_preserves_user_names(self, make_napari_viewer_proxy):
        """Test that user-assigned names are preserved."""
        viewer = make_napari_viewer_proxy()
        layer = viewer.add_shapes(
            [
                [[0, 0], [0, 10], [10, 10], [10, 0]],
                [[20, 20], [20, 30], [30, 30], [30, 20]],
            ],
            shape_type="polygon",
            name="ROIs",
        )
        layer.properties = {"name": ["Arena", ""]}

        RoisWidget(viewer)

        names = list(layer.properties.get("name", []))
        assert names[0] == "Arena"
        assert names[1] == "ROI-1"

    def test_auto_naming_continues_sequence(self, rois_widget_with_layer):
        """Test that auto-naming continues from highest existing number."""
        widget, layer = rois_widget_with_layer

        layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])

        names = list(layer.properties.get("name", []))
        assert names[2] == "ROI-3"
        del widget  # Prevent GC warning

    def test_auto_naming_handles_duplicates_on_add(
        self, make_napari_viewer_proxy
    ):
        """Test that duplicate ROI-<number> names are fixed when adding shapes.

        Duplicate names are NOT resolved during initial linking - they're
        only resolved when shapes are added to ensure new shapes get unique
        names.
        """
        viewer = make_napari_viewer_proxy()
        layer = viewer.add_shapes(
            [[[0, 0], [0, 10], [10, 10], [10, 0]]],
            shape_type="polygon",
            name="ROIs",
        )
        layer.properties = {"name": ["ROI-1"]}

        widget = RoisWidget(viewer)
        layer.add([[20, 20], [20, 30], [30, 30], [30, 20]])

        names = list(layer.properties.get("name", []))
        assert len(names) == 2
        assert names[0] == "ROI-1"
        assert names[1] == "ROI-2"
        del widget  # Prevent GC warning


# ------------------- Tests for table view -----------------------------------#
class TestRoisTableView:
    """Tests for RoisTableView functionality."""

    def test_table_view_selection_syncs_to_layer(self, rois_widget_with_layer):
        """Test that selecting a row in table selects shape in layer."""
        widget, layer = rois_widget_with_layer

        widget.roi_table_view.selectRow(0)

        assert layer.selected_data == {0}

    def test_table_view_allows_name_editing(self, rois_widget_with_layer):
        """Test that name column is editable via double-click."""
        widget, _layer = rois_widget_with_layer

        triggers = widget.roi_table_view.editTriggers()
        assert triggers & QTableView.DoubleClicked


# ------------------- Tests for layer renaming -------------------------------#
class TestLayerRenaming:
    """Tests for layer rename handling."""

    def test_renaming_layer_updates_dropdown(self, rois_widget_with_layer):
        """Test that renaming a layer updates the dropdown."""
        widget, layer = rois_widget_with_layer

        layer.name = "ROIs renamed"

        assert "ROIs renamed" in [
            widget.layer_dropdown.itemText(i)
            for i in range(widget.layer_dropdown.count())
        ]

    def test_renaming_to_roi_pattern_marks_as_roi_layer(
        self, make_napari_viewer_proxy
    ):
        """Test that renaming a layer to ROI pattern marks it as ROI layer."""
        viewer = make_napari_viewer_proxy()
        layer = viewer.add_shapes(name="Other shapes")
        widget = RoisWidget(viewer)

        assert widget.layer_dropdown.currentText() == "Select a layer"

        layer.name = "ROI-Arena"

        assert widget.layer_dropdown.currentText() == "ROI-Arena"
        assert layer.metadata.get("movement_roi_layer") is True


# ------------------- Tests for cleanup --------------------------------------#
class TestWidgetCleanup:
    """Tests for widget cleanup on close."""

    def test_close_event_disconnects_signals(self, rois_widget_with_layer):
        """Test that closing widget disconnects all signals."""
        widget, _layer = rois_widget_with_layer

        with does_not_raise():
            widget.close()

        assert len(widget._connected_layers) == 0

    def test_close_event_clears_model(self, rois_widget_with_layer):
        """Test that closing widget clears the table model."""
        widget, _layer = rois_widget_with_layer

        widget.close()

        assert widget.roi_table_model is None


# ------------------- Tests for model layer deletion -------------------------#
class TestModelLayerDeletion:
    """Tests for RoisTableModel handling of layer deletion."""

    def test_model_handles_layer_deletion(self, rois_widget_with_layer):
        """Test that model handles its layer being deleted."""
        widget, layer = rois_widget_with_layer

        widget.viewer.layers.remove(layer)

        assert widget.roi_table_model is None

    def test_model_ignores_other_layer_deletion(
        self, rois_widget_with_layer, sample_shapes_data
    ):
        """Test that model ignores deletion of unrelated layers."""
        widget, layer = rois_widget_with_layer

        other_layer = widget.viewer.add_shapes(
            sample_shapes_data[:1], name="Other layer"
        )
        widget.viewer.layers.remove(other_layer)

        assert widget.roi_table_model is not None
        assert widget.roi_table_model.layer == layer


# ------------------- Tests for tooltip updates ------------------------------#
class TestTooltipUpdates:
    """Tests for table tooltip updates based on state."""

    def test_tooltip_no_layers(self, rois_widget):
        """Test tooltip when no ROI layers exist."""
        tooltip = rois_widget.roi_table_view.toolTip()
        assert "No ROI layers" in tooltip

    def test_tooltip_empty_layer(self, make_napari_viewer_proxy):
        """Test tooltip when layer has no shapes."""
        viewer = make_napari_viewer_proxy()
        viewer.add_shapes(name="ROIs")
        widget = RoisWidget(viewer)

        tooltip = widget.roi_table_view.toolTip()
        assert "No ROIs in this layer" in tooltip

    def test_tooltip_with_shapes(self, rois_widget_with_layer):
        """Test tooltip when layer has shapes."""
        widget, _layer = rois_widget_with_layer

        tooltip = widget.roi_table_view.toolTip()
        assert "Click a row" in tooltip


# ------------------- Tests for edge cases -----------------------------------#
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_shapes_layer(self, make_napari_viewer_proxy):
        """Test widget handles empty shapes layer."""
        viewer = make_napari_viewer_proxy()
        viewer.add_shapes(name="ROIs")

        with does_not_raise():
            widget = RoisWidget(viewer)

        assert widget.roi_table_model.rowCount() == 0

    def test_model_with_invalid_index(self, rois_widget_with_layer):
        """Test model returns None for invalid index."""
        widget, _layer = rois_widget_with_layer

        invalid_index = widget.roi_table_model.index(99, 0)
        data = widget.roi_table_model.data(invalid_index, Qt.DisplayRole)
        assert data is None

    def test_model_setData_with_invalid_index(self, rois_widget_with_layer):
        """Test setData returns False for invalid index."""
        widget, _layer = rois_widget_with_layer

        invalid_index = widget.roi_table_model.index(99, 0)
        result = widget.roi_table_model.setData(
            invalid_index, "Name", Qt.EditRole
        )

        assert result is False

    def test_model_flags_invalid_index(self, rois_widget_with_layer):
        """Test flags returns NoItemFlags for invalid index."""
        widget, _layer = rois_widget_with_layer

        invalid_index = QModelIndex()
        flags = widget.roi_table_model.flags(invalid_index)

        assert flags == Qt.NoItemFlags

    def test_table_view_selection_with_no_model(self, rois_widget):
        """Test table view handles selection when model is None."""
        with does_not_raise():
            rois_widget.roi_table_view._on_selection_changed(None, None)

    def test_vertical_header_data(self, rois_widget_with_layer):
        """Test vertical header returns row index as string."""
        widget, _layer = rois_widget_with_layer

        header = widget.roi_table_model.headerData(
            0, Qt.Vertical, Qt.DisplayRole
        )
        assert header == "0"

    def test_header_data_non_display_role(self, rois_widget_with_layer):
        """Test headerData returns None for non-display role."""
        widget, _layer = rois_widget_with_layer

        header = widget.roi_table_model.headerData(
            0, Qt.Horizontal, Qt.DecorationRole
        )
        assert header is None
