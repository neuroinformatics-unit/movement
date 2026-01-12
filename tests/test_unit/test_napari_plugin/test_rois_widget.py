"""Unit tests for the ROIs widget in the napari plugin."""

from contextlib import nullcontext as does_not_raise

import pytest
from napari.layers import Shapes
from qtpy.QtCore import QModelIndex, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QTableView,
)

from movement.napari.rois_widget import (
    RoisTableView,
    RoisWidget,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Previous color_by key.*:UserWarning"
)


# ------------------- Fixtures -----------------------------------------------#
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
    """Return a RoisWidget with a viewer and shapes layer containing 2 ROIs."""
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
def test_widget_has_expected_attributes(make_napari_viewer_proxy):
    """Test that the ROIs widget is properly instantiated."""
    viewer = make_napari_viewer_proxy()
    widget = RoisWidget(viewer)
    assert widget.viewer == viewer
    assert widget.roi_table_model is None
    assert isinstance(widget.roi_table_view, RoisTableView)
    assert len(widget._connected_layers) == 0


def test_widget_has_expected_ui_elements(rois_widget):
    """Test that the ROIs widget has all expected UI elements."""
    group_boxes = rois_widget.findChildren(QGroupBox)
    assert len(group_boxes) == 2
    assert rois_widget.findChild(QComboBox) is not None
    assert rois_widget.add_layer_button is not None
    assert rois_widget.findChild(QTableView) is not None


def test_widget_with_custom_colormap(make_napari_viewer_proxy):
    """Test that the widget can be instantiated with a custom colormap."""
    viewer = make_napari_viewer_proxy()
    widget = RoisWidget(viewer, cmap_name="viridis")
    assert widget.color_manager.cmap_name == "viridis"


def test_dropdown_shows_placeholder_when_no_layers(rois_widget):
    """Test dropdown shows placeholder when no ROI layers exist."""
    assert rois_widget.layer_dropdown.currentText() == "Select a layer"


def test_dropdown_populated_with_existing_roi_layer(make_napari_viewer_proxy):
    """Test dropdown is populated when ROI layer exists at init."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="ROIs")
    widget = RoisWidget(viewer)
    assert widget.layer_dropdown.count() == 1
    assert widget.layer_dropdown.currentText() == "ROIs"


# ------------------- Tests for signal/event connections ---------------------#
def test_add_layer_button_connected_to_handler(
    make_napari_viewer_proxy, mocker
):
    """Test that clicking Add new layer button calls the handler."""
    mock_method = mocker.patch(
        "movement.napari.rois_widget.RoisWidget._add_new_layer"
    )
    widget = RoisWidget(make_napari_viewer_proxy())
    widget.add_layer_button.click()
    mock_method.assert_called_once()


def test_dropdown_connected_to_layer_selection_handler(
    make_napari_viewer_proxy, mocker
):
    """Test that changing dropdown selection calls the handler."""
    mock_method = mocker.patch(
        "movement.napari.rois_widget.RoisWidget._on_layer_selected"
    )
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="ROIs")
    viewer.add_shapes(name="ROIs [1]")
    widget = RoisWidget(viewer)

    mock_method.reset_mock()
    widget.layer_dropdown.setCurrentText("ROIs [1]")
    mock_method.assert_called()


def test_layer_added_triggers_dropdown_update(
    make_napari_viewer_proxy, mocker
):
    """Test that adding a layer triggers dropdown update."""
    mock_method = mocker.patch(
        "movement.napari.rois_widget.RoisWidget._update_layer_dropdown"
    )
    viewer = make_napari_viewer_proxy()
    RoisWidget(viewer)

    mock_method.reset_mock()
    viewer.add_shapes(name="ROIs")
    mock_method.assert_called()


def test_layer_removed_triggers_dropdown_update(
    make_napari_viewer_proxy, mocker
):
    """Test that removing a layer triggers dropdown update."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(name="ROIs")
    mock_method = mocker.patch(
        "movement.napari.rois_widget.RoisWidget._update_layer_dropdown"
    )
    _ = RoisWidget(viewer)  # must stay alive to receive signal

    mock_method.reset_mock()
    viewer.layers.remove(layer)
    mock_method.assert_called()


def test_shape_data_change_triggers_model_update(
    rois_widget_with_layer, mocker
):
    """Test that adding a shape triggers model's data change handler."""
    widget, layer = rois_widget_with_layer
    mock_method = mocker.patch.object(
        widget.roi_table_model, "_on_layer_data_changed"
    )
    layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])
    mock_method.assert_called()


# ------------------- Tests for widget methods -------------------------------#
def test_add_new_layer(rois_widget):
    """Test that _add_new_layer creates a properly configured ROI layer."""
    rois_widget._add_new_layer()

    assert len(rois_widget.viewer.layers) == 1
    layer = rois_widget.viewer.layers[0]
    assert isinstance(layer, Shapes)
    assert layer.name.startswith("ROIs")
    assert layer.metadata.get("movement_roi_layer") is True
    assert rois_widget.layer_dropdown.currentText() == layer.name


def test_add_multiple_layers_increments_name(rois_widget):
    """Test that multiple new layers get unique names."""
    rois_widget._add_new_layer()
    rois_widget._add_new_layer()
    layer_names = [layer.name for layer in rois_widget.viewer.layers]
    assert len(set(layer_names)) == 2


def test_update_layer_dropdown_on_layer_added(rois_widget):
    """Test dropdown is updated when a new ROI layer is added."""
    rois_widget.viewer.add_shapes(name="ROIs")
    assert rois_widget.layer_dropdown.count() == 1
    assert rois_widget.layer_dropdown.currentText() == "ROIs"


def test_update_layer_dropdown_on_layer_removed(rois_widget_with_layer):
    """Test dropdown is updated when an ROI layer is removed."""
    widget, layer = rois_widget_with_layer
    assert widget.layer_dropdown.count() == 1

    widget.viewer.layers.remove(layer)
    assert widget.layer_dropdown.currentText() == "Select a layer"


def test_dropdown_ignores_non_roi_layers(make_napari_viewer_proxy):
    """Test dropdown ignores non-ROI shapes layers."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="Other shapes")
    widget = RoisWidget(viewer)
    assert widget.layer_dropdown.currentText() == "Select a layer"


def test_dropdown_includes_layer_with_roi_metadata(make_napari_viewer_proxy):
    """Test dropdown includes layers marked with ROI metadata."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(name="Custom name")
    layer.metadata["movement_roi_layer"] = True
    widget = RoisWidget(viewer)
    assert widget.layer_dropdown.count() == 1
    assert widget.layer_dropdown.currentText() == "Custom name"


def test_dropdown_preserves_selection_on_update(make_napari_viewer_proxy):
    """Test dropdown preserves current selection when updated."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="ROIs")
    viewer.add_shapes(name="ROIs [1]")
    widget = RoisWidget(viewer)
    widget.layer_dropdown.setCurrentText("ROIs [1]")
    viewer.add_shapes(name="ROIs [2]")

    assert widget.layer_dropdown.currentText() == "ROIs [1]"


def test_layer_selection_links_to_model(rois_widget_with_layer):
    """Test that selecting a layer creates a linked table model."""
    widget, layer = rois_widget_with_layer
    assert widget.roi_table_model is not None
    assert widget.roi_table_model.layer == layer


def test_renaming_layer_updates_dropdown(rois_widget_with_layer):
    """Test that renaming a layer updates the dropdown."""
    widget, layer = rois_widget_with_layer
    layer.name = "ROIs renamed"
    assert "ROIs renamed" in [
        widget.layer_dropdown.itemText(i)
        for i in range(widget.layer_dropdown.count())
    ]


def test_renaming_to_roi_pattern_marks_as_roi_layer(make_napari_viewer_proxy):
    """Test that renaming a layer to ROI pattern marks it as ROI layer."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(name="Other shapes")
    widget = RoisWidget(viewer)
    assert widget.layer_dropdown.currentText() == "Select a layer"

    layer.name = "ROI-Arena"
    assert widget.layer_dropdown.currentText() == "ROI-Arena"
    assert layer.metadata.get("movement_roi_layer") is True


def test_close_cleans_up(rois_widget_with_layer):
    """Test that closing widget disconnects signals and clears model."""
    widget, _ = rois_widget_with_layer
    with does_not_raise():
        widget.close()
    assert len(widget._connected_layers) == 0
    assert widget.roi_table_model is None


# ------------------- Tests for ROI auto-naming ------------------------------#
@pytest.mark.parametrize("empty_value", ["", None])
def test_fills_empty_or_none_names(make_napari_viewer_proxy, empty_value):
    """Test that empty/None names are filled with auto-generated names."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(
        [[[0, 0], [0, 10], [10, 10], [10, 0]]],
        shape_type="polygon",
        name="ROIs",
    )
    layer.properties = {"name": [empty_value]}

    RoisWidget(viewer)
    names = list(layer.properties.get("name", []))
    assert names[0] == "ROI-1"


def test_preserves_user_names(make_napari_viewer_proxy):
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


def test_new_shape_gets_auto_name(rois_widget_with_layer):
    """Test that new shapes get auto-assigned names continuing sequence."""
    _, layer = rois_widget_with_layer
    layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])

    names = list(layer.properties.get("name", []))
    assert len(names) == 3
    assert names[2] == "ROI-3"


# ------------------- Tests for RoisTableModel -------------------------------#
def test_model_row_and_column_count(rois_widget_with_layer):
    """Test that model dimensions match the data."""
    widget, _ = rois_widget_with_layer
    assert widget.roi_table_model.rowCount() == 2
    assert widget.roi_table_model.columnCount() == 2


def test_model_header_labels(rois_widget_with_layer):
    """Test that model header labels are correct."""
    widget, _ = rois_widget_with_layer
    assert widget.roi_table_model.headerData(0, Qt.Horizontal) == "Name"
    assert widget.roi_table_model.headerData(1, Qt.Horizontal) == "Shape type"


@pytest.mark.parametrize(
    "column, expected",
    [(0, "ROI-1"), (1, "polygon")],
    ids=["name_column", "shape_type_column"],
)
def test_model_data_returns_correct_values(
    rois_widget_with_layer, column, expected
):
    """Test that model returns correct data for each column."""
    widget, _ = rois_widget_with_layer
    index = widget.roi_table_model.index(0, column)
    assert widget.roi_table_model.data(index, Qt.DisplayRole) == expected


def test_model_setData_updates_roi_name(rois_widget_with_layer):
    """Test that setData updates the ROI name."""
    widget, layer = rois_widget_with_layer
    index = widget.roi_table_model.index(0, 0)
    result = widget.roi_table_model.setData(index, "New Name", Qt.EditRole)

    assert result is True
    assert layer.properties["name"][0] == "New Name"


def test_model_setData_rejects_shape_type_edit(rois_widget_with_layer):
    """Test that setData returns False for shape type column."""
    widget, _ = rois_widget_with_layer
    index = widget.roi_table_model.index(0, 1)
    result = widget.roi_table_model.setData(index, "rectangle", Qt.EditRole)
    assert result is False


@pytest.mark.parametrize(
    "column, is_editable",
    [(0, True), (1, False)],
    ids=["name_editable", "shape_type_not_editable"],
)
def test_model_column_editability(rois_widget_with_layer, column, is_editable):
    """Test that only the name column is editable."""
    widget, _ = rois_widget_with_layer
    index = widget.roi_table_model.index(0, column)
    flags = widget.roi_table_model.flags(index)
    assert bool(flags & Qt.ItemIsEditable) == is_editable


def test_model_updates_on_shape_added(rois_widget_with_layer):
    """Test that adding a shape updates the model."""
    widget, layer = rois_widget_with_layer
    initial_count = widget.roi_table_model.rowCount()
    layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])

    assert widget.roi_table_model.rowCount() == initial_count + 1


def test_model_updates_on_shape_removed(rois_widget_with_layer):
    """Test that removing a shape updates the model."""
    widget, layer = rois_widget_with_layer
    initial_count = widget.roi_table_model.rowCount()
    layer.selected_data = {0}
    layer.remove_selected()

    assert widget.roi_table_model.rowCount() == initial_count - 1


def test_model_cleared_on_layer_deletion(rois_widget_with_layer):
    """Test that deleting the layer clears the model."""
    widget, layer = rois_widget_with_layer
    widget.viewer.layers.remove(layer)
    assert widget.roi_table_model is None


def test_model_ignores_other_layer_deletion(
    rois_widget_with_layer, sample_shapes_data
):
    """Test that model ignores deletion of unrelated layers."""
    widget, layer = rois_widget_with_layer
    other_layer = widget.viewer.add_shapes(
        sample_shapes_data[:1], name="Other layer"
    )
    widget.viewer.layers.remove(other_layer)

    assert widget.roi_table_model is not None
    assert widget.roi_table_model.layer == layer


# ------------------- Tests for RoisTableView --------------------------------#
def test_table_selection_syncs_to_layer(rois_widget_with_layer):
    """Test that selecting a row in table selects shape in layer."""
    widget, layer = rois_widget_with_layer
    widget.roi_table_view.selectRow(0)
    assert layer.selected_data == {0}


def test_table_allows_name_editing(rois_widget_with_layer):
    """Test that name column is editable via double-click."""
    widget, _ = rois_widget_with_layer
    triggers = widget.roi_table_view.editTriggers()
    assert triggers & QTableView.DoubleClicked


# ------------------- Tests for tooltips -------------------------------------#
def test_tooltip_no_layers(rois_widget):
    """Test tooltip when no ROI layers exist."""
    tooltip = rois_widget.roi_table_view.toolTip()
    assert "No ROI layers" in tooltip


def test_tooltip_empty_layer(make_napari_viewer_proxy):
    """Test tooltip when layer has no shapes."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="ROIs")
    widget = RoisWidget(viewer)
    tooltip = widget.roi_table_view.toolTip()
    assert "No ROIs in this layer" in tooltip


def test_tooltip_with_shapes(rois_widget_with_layer):
    """Test tooltip when layer has shapes."""
    widget, _ = rois_widget_with_layer
    tooltip = widget.roi_table_view.toolTip()
    assert "Click a row" in tooltip


# ------------------- Tests for edge cases -----------------------------------#
def test_empty_shapes_layer(make_napari_viewer_proxy):
    """Test widget handles empty shapes layer."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="ROIs")
    with does_not_raise():
        widget = RoisWidget(viewer)

    assert widget.roi_table_model.rowCount() == 0


@pytest.mark.parametrize(
    "method, args, expected",
    [
        ("data", (Qt.DisplayRole,), None),
        ("setData", ("Name", Qt.EditRole), False),
    ],
    ids=["data_returns_none", "setData_returns_false"],
)
def test_model_with_invalid_row_index(
    rois_widget_with_layer, method, args, expected
):
    """Test model methods return appropriate values for invalid index."""
    widget, _ = rois_widget_with_layer
    invalid_index = widget.roi_table_model.index(99, 0)
    result = getattr(widget.roi_table_model, method)(invalid_index, *args)
    assert result == expected


def test_model_flags_invalid_index(rois_widget_with_layer):
    """Test flags returns NoItemFlags for invalid index."""
    widget, _ = rois_widget_with_layer
    invalid_index = QModelIndex()
    flags = widget.roi_table_model.flags(invalid_index)
    assert flags == Qt.NoItemFlags


def test_table_view_selection_with_no_model(rois_widget):
    """Test table view handles selection when model is None."""
    with does_not_raise():
        rois_widget.roi_table_view._on_selection_changed(None, None)


@pytest.mark.parametrize(
    "orientation, role, expected",
    [
        (Qt.Vertical, Qt.DisplayRole, "0"),
        (Qt.Horizontal, Qt.DecorationRole, None),
    ],
    ids=["vertical_header", "non_display_role"],
)
def test_model_header_data_edge_cases(
    rois_widget_with_layer, orientation, role, expected
):
    """Test headerData for vertical orientation and non-display roles."""
    widget, _ = rois_widget_with_layer
    header = widget.roi_table_model.headerData(0, orientation, role)
    assert header == expected
