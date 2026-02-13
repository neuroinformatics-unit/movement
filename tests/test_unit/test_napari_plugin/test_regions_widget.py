"""Unit tests for the Regions widget in the napari plugin."""

from contextlib import nullcontext as does_not_raise

import pytest
from napari.layers import Shapes
from qtpy.QtCore import QItemSelection, QModelIndex, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QTableView,
)

from movement.napari.regions_widget import (
    DEFAULT_REGION_NAME,
    RegionsTableView,
    RegionsWidget,
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
def regions_widget(make_napari_viewer_proxy):
    """Return a RegionsWidget with an empty viewer."""
    viewer = make_napari_viewer_proxy()
    return RegionsWidget(viewer)


@pytest.fixture
def regions_widget_with_layer(make_napari_viewer_proxy, sample_shapes_data):
    """Return a RegionsWidget with a viewer and shapes layer with 2 regions."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(
        sample_shapes_data[:2],
        shape_type="polygon",
        name="Regions",
    )
    layer.properties = {"name": [DEFAULT_REGION_NAME, DEFAULT_REGION_NAME]}
    widget = RegionsWidget(viewer)
    return widget, layer


# ------------------- Tests for widget instantiation -------------------------#
def test_widget_has_expected_attributes(make_napari_viewer_proxy):
    """Test that the Regions widget is properly instantiated."""
    viewer = make_napari_viewer_proxy()
    widget = RegionsWidget(viewer)
    assert widget.viewer == viewer
    assert widget.region_table_model is None
    assert isinstance(widget.region_table_view, RegionsTableView)
    assert len(widget._connected_layers) == 0


def test_widget_has_expected_ui_elements(regions_widget):
    """Test that the Regions widget has all expected UI elements."""
    group_boxes = regions_widget.findChildren(QGroupBox)
    assert len(group_boxes) == 2
    assert regions_widget.findChild(QComboBox) is not None
    assert regions_widget.add_layer_button is not None
    assert regions_widget.findChild(QTableView) is not None


def test_widget_with_custom_colormap(make_napari_viewer_proxy):
    """Test that the widget can be instantiated with a custom colormap."""
    viewer = make_napari_viewer_proxy()
    widget = RegionsWidget(viewer, cmap_name="viridis")
    assert widget.color_manager.cmap_name == "viridis"


def test_dropdown_shows_placeholder_when_no_layers(regions_widget):
    """Test dropdown shows placeholder when no region layers exist."""
    assert regions_widget.layer_dropdown.currentText() == "Select a layer"


def test_dropdown_populated_with_existing_region_layer(
    make_napari_viewer_proxy,
):
    """Test dropdown is populated when region layer exists at init."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="Regions")
    widget = RegionsWidget(viewer)
    assert widget.layer_dropdown.count() == 1
    assert widget.layer_dropdown.currentText() == "Regions"


def test_auto_assign_names_pads_missing_name_property(
    make_napari_viewer_proxy, sample_shapes_data
):
    """Test that missing name property gets created and filled."""
    viewer = make_napari_viewer_proxy()
    # Create layer with shapes but no "name" property
    layer = viewer.add_shapes(sample_shapes_data[:2], name="Regions")
    # Creating widget triggers _auto_assign_region_names which pads names
    RegionsWidget(viewer)
    # Names should be created and filled to match shape count
    assert len(layer.properties["name"]) == 2


# ------------------- Tests for signal/event connections ---------------------#
def test_add_layer_button_connected_to_handler(
    make_napari_viewer_proxy, mocker
):
    """Test that clicking Add new layer button calls the handler."""
    mock_method = mocker.patch(
        "movement.napari.regions_widget.RegionsWidget._add_new_layer"
    )
    widget = RegionsWidget(make_napari_viewer_proxy())
    widget.add_layer_button.click()
    mock_method.assert_called_once()


def test_dropdown_connected_to_layer_selection_handler(
    make_napari_viewer_proxy, mocker
):
    """Test that changing dropdown selection calls the handler."""
    mock_method = mocker.patch(
        "movement.napari.regions_widget.RegionsWidget._on_layer_selected"
    )
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="Regions")
    viewer.add_shapes(name="Regions [1]")
    widget = RegionsWidget(viewer)

    mock_method.reset_mock()
    widget.layer_dropdown.setCurrentText("Regions [1]")
    mock_method.assert_called()


def test_layer_added_triggers_dropdown_update(
    make_napari_viewer_proxy, mocker
):
    """Test that adding a layer triggers dropdown update."""
    mock_method = mocker.patch(
        "movement.napari.regions_widget.RegionsWidget._update_layer_dropdown"
    )
    viewer = make_napari_viewer_proxy()
    RegionsWidget(viewer)

    mock_method.reset_mock()
    viewer.add_shapes(name="Regions")
    mock_method.assert_called()


def test_layer_removed_triggers_dropdown_update(
    make_napari_viewer_proxy, mocker
):
    """Test that removing a layer triggers dropdown update."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(name="Regions")
    mock_method = mocker.patch(
        "movement.napari.regions_widget.RegionsWidget._update_layer_dropdown"
    )
    _ = RegionsWidget(viewer)  # must stay alive to receive signal

    mock_method.reset_mock()
    viewer.layers.remove(layer)
    mock_method.assert_called()


def test_shape_data_change_triggers_model_update(
    regions_widget_with_layer, mocker
):
    """Test that adding a shape triggers model's data change handler."""
    widget, layer = regions_widget_with_layer
    mock_method = mocker.patch.object(
        widget.region_table_model, "_on_layer_data_changed"
    )
    layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])
    mock_method.assert_called()


def test_set_data_event_triggers_handler(regions_widget_with_layer, mocker):
    """Test that set_data event triggers the handler."""
    widget, layer = regions_widget_with_layer
    mock_method = mocker.patch.object(
        widget.region_table_model, "_on_layer_set_data"
    )
    layer.events.set_data()
    mock_method.assert_called()


# ------------------- Tests for widget methods -------------------------------#
def test_add_new_layer(regions_widget):
    """Test that _add_new_layer creates a properly configured region layer."""
    regions_widget._add_new_layer()

    assert len(regions_widget.viewer.layers) == 1
    layer = regions_widget.viewer.layers[0]
    assert isinstance(layer, Shapes)
    assert layer.name.startswith("Regions")
    assert layer.metadata.get("movement_region_layer") is True
    assert regions_widget.layer_dropdown.currentText() == layer.name


def test_add_multiple_layers_increments_name(regions_widget):
    """Test that multiple new layers get unique names."""
    regions_widget._add_new_layer()
    regions_widget._add_new_layer()
    layer_names = [layer.name for layer in regions_widget.viewer.layers]
    assert len(set(layer_names)) == 2


def test_update_layer_dropdown_on_layer_added(regions_widget):
    """Test dropdown is updated when a new region layer is added."""
    regions_widget.viewer.add_shapes(name="Regions")
    assert regions_widget.layer_dropdown.count() == 1
    assert regions_widget.layer_dropdown.currentText() == "Regions"


def test_update_layer_dropdown_on_layer_removed(regions_widget_with_layer):
    """Test dropdown is updated when a region layer is removed."""
    widget, layer = regions_widget_with_layer
    assert widget.layer_dropdown.count() == 1

    widget.viewer.layers.remove(layer)
    assert widget.layer_dropdown.currentText() == "Select a layer"


def test_dropdown_ignores_non_region_layers(make_napari_viewer_proxy):
    """Test dropdown ignores non-region shapes layers."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="Other shapes")
    widget = RegionsWidget(viewer)
    assert widget.layer_dropdown.currentText() == "Select a layer"


def test_dropdown_includes_layer_with_region_metadata(
    make_napari_viewer_proxy,
):
    """Test dropdown includes layers marked with region metadata."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(name="Custom name")
    layer.metadata["movement_region_layer"] = True
    widget = RegionsWidget(viewer)
    assert widget.layer_dropdown.count() == 1
    assert widget.layer_dropdown.currentText() == "Custom name"


def test_dropdown_preserves_selection_on_update(make_napari_viewer_proxy):
    """Test dropdown preserves current selection when updated."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="Regions")
    viewer.add_shapes(name="Regions [1]")
    widget = RegionsWidget(viewer)
    widget.layer_dropdown.setCurrentText("Regions [1]")
    viewer.add_shapes(name="Regions [2]")

    assert widget.layer_dropdown.currentText() == "Regions [1]"


def test_layer_selection_links_to_model(regions_widget_with_layer):
    """Test that selecting a layer creates a linked table model."""
    widget, layer = regions_widget_with_layer
    assert widget.region_table_model is not None
    assert widget.region_table_model.layer == layer


def test_renaming_layer_updates_dropdown(regions_widget_with_layer):
    """Test that renaming a layer updates the dropdown."""
    widget, layer = regions_widget_with_layer
    layer.name = "Regions renamed"
    assert "Regions renamed" in [
        widget.layer_dropdown.itemText(i)
        for i in range(widget.layer_dropdown.count())
    ]


def test_renaming_to_region_pattern_marks_as_region_layer(
    make_napari_viewer_proxy,
):
    """Test that renaming to Region pattern marks it as a region layer."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(name="Other shapes")
    widget = RegionsWidget(viewer)
    assert widget.layer_dropdown.currentText() == "Select a layer"

    layer.name = "Region-Arena"
    assert widget.layer_dropdown.currentText() == "Region-Arena"
    assert layer.metadata.get("movement_region_layer") is True


def test_close_cleans_up(regions_widget_with_layer):
    """Test that closing widget disconnects signals and clears model."""
    widget, _ = regions_widget_with_layer
    with does_not_raise():
        widget.close()
    assert len(widget._connected_layers) == 0
    assert widget.region_table_model is None


# ------------------- Tests for region auto-naming ---------------------------#
@pytest.mark.parametrize("empty_value", ["", None])
def test_fills_empty_or_none_names(make_napari_viewer_proxy, empty_value):
    """Test that empty/None names are filled with default name."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(
        [[[0, 0], [0, 10], [10, 10], [10, 0]]],
        shape_type="polygon",
        name="Regions",
    )
    layer.properties = {"name": [empty_value]}

    RegionsWidget(viewer)
    names = list(layer.properties.get("name", []))
    assert names[0] == DEFAULT_REGION_NAME


def test_preserves_user_names(make_napari_viewer_proxy):
    """Test that user-assigned names are preserved."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(
        [
            [[0, 0], [0, 10], [10, 10], [10, 0]],
            [[20, 20], [20, 30], [30, 30], [30, 20]],
        ],
        shape_type="polygon",
        name="Regions",
    )
    layer.properties = {"name": ["Arena", ""]}

    RegionsWidget(viewer)
    names = list(layer.properties.get("name", []))
    assert names[0] == "Arena"
    assert names[1] == DEFAULT_REGION_NAME


def test_new_drawn_shape_gets_default_name(regions_widget_with_layer):
    """Test that newly drawn shapes get the default name."""
    _, layer = regions_widget_with_layer
    layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])

    names = list(layer.properties.get("name", []))
    assert len(names) == 3
    # Drawn shapes get default name (layer.add emits "added" event)
    assert names[2] == DEFAULT_REGION_NAME


# ------------------- Tests for RegionsTableModel ----------------------------#
def test_model_row_and_column_count(regions_widget_with_layer):
    """Test that model dimensions match the data."""
    widget, _ = regions_widget_with_layer
    assert widget.region_table_model.rowCount() == 2
    assert widget.region_table_model.columnCount() == 2


def test_model_header_labels(regions_widget_with_layer):
    """Test that model header labels are correct."""
    widget, _ = regions_widget_with_layer
    assert widget.region_table_model.headerData(0, Qt.Horizontal) == "Name"
    assert (
        widget.region_table_model.headerData(1, Qt.Horizontal) == "Shape type"
    )


@pytest.mark.parametrize(
    "row, column, role, expected",
    [
        (0, 0, Qt.DisplayRole, DEFAULT_REGION_NAME),
        (0, 1, Qt.DisplayRole, "polygon"),
        (0, 0, Qt.EditRole, DEFAULT_REGION_NAME),
        (0, 1, Qt.EditRole, None),  # EditRole only supported for col 0
    ],
    ids=["display_name", "display_shape", "edit_name", "edit_shape_col"],
)
def test_model_data_returns_correct_values(
    regions_widget_with_layer, row, column, role, expected
):
    """Test that model returns correct data for each column and role."""
    widget, _ = regions_widget_with_layer
    index = widget.region_table_model.index(row, column)
    assert widget.region_table_model.data(index, role) == expected


def test_model_data_returns_none_for_stale_index(regions_widget_with_layer):
    """Test that data returns None when index row exceeds layer data."""
    widget, layer = regions_widget_with_layer
    # Get valid index for row 1 (layer has 2 shapes)
    index = widget.region_table_model.index(1, 0)
    # Remove all shapes, making the index stale
    layer.data = []
    # Index is still structurally valid but row >= len(layer.data)
    assert widget.region_table_model.data(index, Qt.DisplayRole) is None


def test_model_setData_updates_region_name(regions_widget_with_layer):
    """Test that setData updates the region name."""
    widget, layer = regions_widget_with_layer
    index = widget.region_table_model.index(0, 0)
    result = widget.region_table_model.setData(index, "New Name", Qt.EditRole)

    assert result is True
    assert layer.properties["name"][0] == "New Name"


def test_model_setData_rejects_shape_type_edit(regions_widget_with_layer):
    """Test that setData returns False for shape type column."""
    widget, _ = regions_widget_with_layer
    index = widget.region_table_model.index(0, 1)
    result = widget.region_table_model.setData(index, "rectangle", Qt.EditRole)
    assert result is False


def test_model_setData_rejects_stale_index(regions_widget_with_layer):
    """Test that setData returns False when index row exceeds layer data."""
    widget, layer = regions_widget_with_layer
    index = widget.region_table_model.index(1, 0)
    layer.data = []
    result = widget.region_table_model.setData(index, "Name", Qt.EditRole)
    assert result is False


@pytest.mark.parametrize(
    "column, is_editable",
    [(0, True), (1, False)],
    ids=["name_editable", "shape_type_not_editable"],
)
def test_model_column_editability(
    regions_widget_with_layer, column, is_editable
):
    """Test that only the name column is editable."""
    widget, _ = regions_widget_with_layer
    index = widget.region_table_model.index(0, column)
    flags = widget.region_table_model.flags(index)
    assert bool(flags & Qt.ItemIsEditable) == is_editable


def test_model_updates_on_shape_added(regions_widget_with_layer):
    """Test that adding a shape updates the model."""
    widget, layer = regions_widget_with_layer
    initial_count = widget.region_table_model.rowCount()
    layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])

    assert widget.region_table_model.rowCount() == initial_count + 1


def test_sync_names_assigns_default_to_new_shapes(regions_widget_with_layer):
    """Test that _sync_names_on_shape_change assigns default name to new."""
    widget, layer = regions_widget_with_layer
    model = widget.region_table_model
    # Add a shape so layer has 3 shapes
    layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])
    # Reset _last_shape_count to simulate state before shape was added
    model._last_shape_count = 2
    # Call sync with assign_default_to_new=True
    model._sync_names_on_shape_change(n_shapes=3, assign_default_to_new=True)
    # New shape should have default name
    assert layer.properties["name"][2] == DEFAULT_REGION_NAME


def test_model_updates_on_shape_removed(regions_widget_with_layer):
    """Test that removing a shape updates the model."""
    widget, layer = regions_widget_with_layer
    initial_count = widget.region_table_model.rowCount()
    layer.selected_data = {0}
    layer.remove_selected()

    assert widget.region_table_model.rowCount() == initial_count - 1


def test_set_data_handler_updates_model_and_preserves_names(
    regions_widget_with_layer,
):
    """Test that _on_layer_set_data updates model and preserves names.

    This handler is triggered by copy-paste operations. It should detect
    shape count changes and update the model without overwriting names.
    """
    widget, layer = regions_widget_with_layer
    model = widget.region_table_model

    # Add a third shape with a custom copied name directly to the layer
    layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])
    layer.properties = {"name": ["Region-A", "Region-B", "Region-B"]}

    # Simulate the state before a "paste" by resetting the shape count tracker
    model._last_shape_count = 2

    # Call the handler (as would happen on set_data event)
    model._on_layer_set_data()

    # Verify model updated and all names preserved
    assert model.rowCount() == 3
    assert model._last_shape_count == 3
    expected_names = ["Region-A", "Region-B", "Region-B"]
    assert list(layer.properties["name"]) == expected_names


def test_model_cleared_on_layer_deletion(regions_widget_with_layer):
    """Test that deleting the layer clears the model."""
    widget, layer = regions_widget_with_layer
    widget.viewer.layers.remove(layer)
    assert widget.region_table_model is None


def test_model_ignores_other_layer_deletion(
    regions_widget_with_layer, sample_shapes_data
):
    """Test that model ignores deletion of unrelated layers."""
    widget, layer = regions_widget_with_layer
    other_layer = widget.viewer.add_shapes(
        sample_shapes_data[:1], name="Other layer"
    )
    widget.viewer.layers.remove(other_layer)

    assert widget.region_table_model is not None
    assert widget.region_table_model.layer == layer


@pytest.mark.parametrize(
    "method_name",
    ["_on_layer_data_changed", "_on_layer_set_data"],
)
def test_layer_event_handlers_return_early_when_no_layer(
    regions_widget_with_layer, method_name
):
    """Test that layer event handlers return early when layer is None."""
    widget, _ = regions_widget_with_layer
    model = widget.region_table_model
    model.layer = None
    method = getattr(model, method_name)
    with does_not_raise():
        method(event=None)


def test_on_layer_deleted_cleans_up_model(regions_widget_with_layer, mocker):
    """Test that _on_layer_deleted disconnects and clears the model."""
    widget, layer = regions_widget_with_layer
    model = widget.region_table_model
    # Create mock event with the layer as value
    mock_event = mocker.Mock()
    mock_event.value = layer
    # Call _on_layer_deleted directly
    model._on_layer_deleted(mock_event)
    # Model should have cleared its layer reference
    assert model.layer is None


# ------------------- Tests for RegionsTableView -----------------------------#
def test_table_selection_syncs_to_layer(regions_widget_with_layer):
    """Test that selecting a row in table selects shape in layer."""
    widget, layer = regions_widget_with_layer
    widget.region_table_view.selectRow(0)
    assert layer.selected_data == {0}


def test_table_allows_name_editing(regions_widget_with_layer):
    """Test that name column is editable via double-click."""
    widget, _ = regions_widget_with_layer
    triggers = widget.region_table_view.editTriggers()
    assert triggers & QTableView.DoubleClicked


# ------------------- Tests for tooltips -------------------------------------#
def test_tooltip_no_layers(regions_widget):
    """Test tooltip when no region layers exist."""
    tooltip = regions_widget.region_table_view.toolTip()
    assert "No region layers" in tooltip


def test_tooltip_empty_layer(make_napari_viewer_proxy):
    """Test tooltip when layer has no shapes."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="Regions")
    widget = RegionsWidget(viewer)
    tooltip = widget.region_table_view.toolTip()
    assert "No regions in this layer" in tooltip


def test_tooltip_with_shapes(regions_widget_with_layer):
    """Test tooltip when layer has shapes."""
    widget, _ = regions_widget_with_layer
    tooltip = widget.region_table_view.toolTip()
    assert "Click a row" in tooltip


# ------------------- Tests for edge cases -----------------------------------#
def test_empty_shapes_layer(make_napari_viewer_proxy):
    """Test widget handles empty shapes layer."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="Regions")
    with does_not_raise():
        widget = RegionsWidget(viewer)

    assert widget.region_table_model.rowCount() == 0


@pytest.mark.parametrize(
    "method, args, expected",
    [
        ("data", (Qt.DisplayRole,), None),
        ("setData", ("Name", Qt.EditRole), False),
    ],
    ids=["data_returns_none", "setData_returns_false"],
)
def test_model_with_invalid_row_index(
    regions_widget_with_layer, method, args, expected
):
    """Test model methods return appropriate values for invalid index."""
    widget, _ = regions_widget_with_layer
    invalid_index = widget.region_table_model.index(99, 0)
    result = getattr(widget.region_table_model, method)(invalid_index, *args)
    assert result == expected


def test_model_flags_invalid_index(regions_widget_with_layer):
    """Test flags returns NoItemFlags for invalid index."""
    widget, _ = regions_widget_with_layer
    invalid_index = QModelIndex()
    flags = widget.region_table_model.flags(invalid_index)
    assert flags == Qt.NoItemFlags


def test_table_view_selection_with_no_model(regions_widget):
    """Test table view handles selection when model is None."""
    with does_not_raise():
        regions_widget.region_table_view._on_selection_changed(None, None)


def test_table_view_selection_with_empty_indexes(regions_widget_with_layer):
    """Test table view handles empty selection indexes."""
    widget, _ = regions_widget_with_layer
    empty_selection = QItemSelection()
    with does_not_raise():
        widget.region_table_view._on_selection_changed(empty_selection, None)


@pytest.mark.parametrize(
    "orientation, role, expected",
    [
        (Qt.Vertical, Qt.DisplayRole, "0"),
        (Qt.Horizontal, Qt.DecorationRole, None),
    ],
    ids=["vertical_header", "non_display_role"],
)
def test_model_header_data_edge_cases(
    regions_widget_with_layer, orientation, role, expected
):
    """Test headerData for vertical orientation and non-display roles."""
    widget, _ = regions_widget_with_layer
    header = widget.region_table_model.headerData(0, orientation, role)
    assert header == expected
