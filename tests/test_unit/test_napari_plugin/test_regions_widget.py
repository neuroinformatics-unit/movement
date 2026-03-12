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
    DROPDOWN_PLACEHOLDER,
    RegionsTableView,
    RegionsWidget,
    _unique_name,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Previous color_by key.*:UserWarning"
)


# ------------------- Helpers ------------------------------------------------#
def add_regions_layer(viewer, data=None, name="Regions", **kwargs):
    """Add a shapes layer marked as a movement region layer to a viewer."""
    return viewer.add_shapes(
        data,
        name=name,
        metadata={"movement_regions_layer": True},
        **kwargs,
    )


# ------------------- Fixtures -----------------------------------------------#
@pytest.fixture
def two_polygons():
    """Return data for 2 sample polygon shapes."""
    return [
        [[0, 0], [0, 10], [10, 10], [10, 0]],
        [[20, 20], [20, 30], [30, 30], [30, 20]],
    ]


@pytest.fixture
def regions_widget(make_napari_viewer_proxy):
    """Return a viewer with a Regions widget."""
    viewer = make_napari_viewer_proxy()
    return RegionsWidget(viewer)


@pytest.fixture
def regions_widget_with_layer(regions_widget, two_polygons):
    """Return a RegionsWidget and a shapes layer with 2 regions."""
    viewer = regions_widget.viewer
    layer = add_regions_layer(viewer, two_polygons, shape_type="polygon")
    layer.properties = {"name": [DEFAULT_REGION_NAME, DEFAULT_REGION_NAME]}
    return regions_widget, layer


# ------------------- Tests for widget instantiation -------------------------#
def test_widget_has_expected_attributes(make_napari_viewer_proxy):
    """Test that the Regions widget is properly instantiated."""
    viewer = make_napari_viewer_proxy()
    widget = RegionsWidget(viewer)
    assert widget.viewer == viewer
    assert widget.region_table_model is None
    assert isinstance(widget.region_table_view, RegionsTableView)


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
    assert regions_widget.layer_dropdown.currentText() == DROPDOWN_PLACEHOLDER
    assert not regions_widget.layer_dropdown.model().item(0).isEnabled()


def test_dropdown_populated_with_existing_region_layer(
    make_napari_viewer_proxy,
):
    """Test dropdown is populated when region layer exists at init."""
    viewer = make_napari_viewer_proxy()
    add_regions_layer(viewer)
    widget = RegionsWidget(viewer)
    assert widget.layer_dropdown.count() == 1
    assert widget.layer_dropdown.currentText() == "Regions"


def test_auto_assign_names_pads_missing_name_property(
    make_napari_viewer_proxy, two_polygons
):
    """Test that missing region name property gets created and filled."""
    viewer = make_napari_viewer_proxy()
    # Create layer with shapes but no "name" assigned to each shape
    # (Note that "Regions" is the name of the layer)
    layer = add_regions_layer(viewer, two_polygons)
    # Creating widget triggers _auto_assign_region_names which pads
    # the list of region names to match the number of shapes in the layer
    RegionsWidget(viewer)
    # Region names should be created, filled, and made unique
    names = layer.properties["name"]
    assert len(names) == 2
    assert all(name.startswith(DEFAULT_REGION_NAME) for name in names)
    assert len(set(names)) == 2  # all unique


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
    add_regions_layer(viewer)

    # Create second layer to switch to
    add_regions_layer(viewer, name="Regions [1]")
    widget = RegionsWidget(viewer)

    # Reset calls to mock since the widget initialisation
    # triggers `_on_layer_selected` internally
    # (when the dropdown is populated and a layer is auto-selected)
    mock_method.reset_mock()
    widget.layer_dropdown.setCurrentText("Regions [1]")
    mock_method.assert_called_once()


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
    mock_method.assert_called_once()


def test_layer_removed_triggers_dropdown_update(
    make_napari_viewer_proxy, mocker
):
    """Test that removing a layer triggers dropdown update."""
    mock_method = mocker.patch(
        "movement.napari.regions_widget.RegionsWidget._update_layer_dropdown"
    )
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(name="Regions")

    # _ to avoid it being gc (must stay alive to receive signal)
    _ = RegionsWidget(viewer)

    mock_method.reset_mock()
    viewer.layers.remove(layer)
    mock_method.assert_called_once()


def test_shape_data_change_triggers_model_update(
    regions_widget_with_layer, mocker
):
    """Test that adding a shape to an existing layer triggers
    table model's data change handler.
    """
    widget, layer = regions_widget_with_layer
    mock_method = mocker.patch.object(
        widget.region_table_model, "_on_layer_data_changed"
    )
    layer.add([[60, 60], [60, 70], [70, 70], [70, 60]])
    mock_method.assert_called()


def test_set_data_event_triggers_handler(regions_widget_with_layer, mocker):
    """Test that assigning to layer.data fires the set_data event
    and calls the handler.
    """
    widget, layer = regions_widget_with_layer
    mock_method = mocker.patch.object(
        widget.region_table_model, "_on_layer_set_data"
    )
    layer.data = [shape * 2 for shape in layer.data]  # scale existing shapes
    mock_method.assert_called()  # napari fires set_data multiple times


# ------------------- Tests for widget methods -------------------------------#
def test_add_new_layer(regions_widget):
    """Test that _add_new_layer creates a properly configured region layer."""
    regions_widget._add_new_layer()

    assert len(regions_widget.viewer.layers) == 1
    layer = regions_widget.viewer.layers[0]
    assert isinstance(layer, Shapes)
    assert layer.name.startswith("Regions")
    assert layer.metadata.get("movement_regions_layer") is True
    assert regions_widget.layer_dropdown.currentText() == layer.name


def test_add_multiple_layers_increments_name(regions_widget):
    """Test that multiple new layers get unique names."""
    regions_widget._add_new_layer()
    regions_widget._add_new_layer()
    layer_names = [layer.name for layer in regions_widget.viewer.layers]
    assert len(set(layer_names)) == 2


def test_update_layer_dropdown_on_layer_added(regions_widget):
    """Test dropdown is updated when a new region layer is added."""
    add_regions_layer(regions_widget.viewer)
    assert regions_widget.layer_dropdown.count() == 1
    assert regions_widget.layer_dropdown.currentText() == "Regions"


def test_update_layer_dropdown_on_layer_removed(regions_widget_with_layer):
    """Test dropdown is updated when a region layer is removed."""
    widget, layer = regions_widget_with_layer
    assert widget.layer_dropdown.count() == 1

    widget.viewer.layers.remove(layer)
    assert widget.layer_dropdown.currentText() == DROPDOWN_PLACEHOLDER


def test_dropdown_ignores_non_region_layers(make_napari_viewer_proxy):
    """Test dropdown ignores non-region shapes layers."""
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(name="Other shapes")
    widget = RegionsWidget(viewer)
    assert widget.layer_dropdown.currentText() == DROPDOWN_PLACEHOLDER
    # the dropdown count should hold placeholder text only
    assert widget.layer_dropdown.count() == 1


def test_dropdown_includes_layer_with_region_metadata(
    make_napari_viewer_proxy,
):
    """Test dropdown includes layers marked with region metadata."""
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(name="Custom name")
    layer.metadata["movement_regions_layer"] = True
    widget = RegionsWidget(viewer)
    assert widget.layer_dropdown.count() == 1
    assert widget.layer_dropdown.currentText() == "Custom name"


def test_dropdown_follows_napari_when_new_region_layer_added(
    make_napari_viewer_proxy,
):
    """Test dropdown follows napari's active layer when a new region is added.

    When napari adds a new region layer, it becomes the active layer, so
    the dropdown syncs to it (bidirectional layer selection).
    """
    viewer = make_napari_viewer_proxy()
    add_regions_layer(viewer)
    widget = RegionsWidget(viewer)
    widget.layer_dropdown.setCurrentText("Regions")

    add_regions_layer(viewer, name="Regions [1]")
    # napari makes the newly added layer active, so the dropdown follows
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
    assert widget.layer_dropdown.findText("Regions renamed") != -1
    # findText returns the index of the matching item or -1 if not found


def test_close_cleans_up(regions_widget_with_layer):
    """Test that closing widget disconnects signals and clears model."""
    widget, _ = regions_widget_with_layer
    with does_not_raise():
        widget.close()
    assert widget.region_table_model is None


# ------------------- Tests for _unique_name ---------------------------------#
@pytest.mark.parametrize(
    "base, existing, expected",
    [
        pytest.param("Region", [], "Region", id="no_conflict"),
        pytest.param("Region", ["Region"], "Region [1]", id="one_conflict"),
        pytest.param(
            "Region",
            ["Region", "Region [1]"],
            "Region [2]",
            id="two_conflicts",
        ),
        pytest.param(
            "Region [1]",
            ["Region", "Region [1]"],
            "Region [2]",
            id="suffix_stripped_before_search",
        ),
        pytest.param(
            "Region [1]",
            ["Region", "Region [1]", "Region [2]"],
            "Region [3]",
            id="suffix_stripped_counts_up",
        ),
    ],
)
def test_unique_name(base, existing, expected):
    """Test that _unique_name returns a unique name."""
    assert _unique_name(base, existing) == expected


# ------------------- Tests for region auto-naming ---------------------------#
@pytest.mark.parametrize("empty_value", ["", None])
def test_fills_empty_or_none_names(
    make_napari_viewer_proxy, two_polygons, empty_value
):
    """Test that empty/None names are filled with default name."""
    viewer = make_napari_viewer_proxy()
    layer = add_regions_layer(viewer, two_polygons[:1], shape_type="polygon")
    layer.properties = {"name": [empty_value]}

    RegionsWidget(viewer)
    assert layer.properties["name"][0] == DEFAULT_REGION_NAME


def test_preserves_user_names(make_napari_viewer_proxy, two_polygons):
    """Test that user-assigned names are preserved."""
    viewer = make_napari_viewer_proxy()
    layer = add_regions_layer(viewer, two_polygons, shape_type="polygon")
    layer.properties = {"name": ["Arena", ""]}

    RegionsWidget(viewer)
    assert all(layer.properties["name"] == ["Arena", DEFAULT_REGION_NAME])


def test_new_drawn_shape_gets_default_name(
    regions_widget_with_layer, two_polygons
):
    """Test that newly drawn shapes get a unique default name."""
    _, layer = regions_widget_with_layer
    layer.add(two_polygons[:1])

    names = layer.properties["name"]
    assert len(names) == 3
    # The last drawn shape gets a unique name derived from DEFAULT_REGION_NAME
    assert names[-1].startswith(DEFAULT_REGION_NAME)
    assert names[-1] not in names[:-1]  # new name is unique among prior names


# ------------------- Tests for RegionsTableModel ----------------------------#
def test_table_model_row_and_column_count(regions_widget_with_layer):
    """Test that table model dimensions match the data."""
    widget, _ = regions_widget_with_layer
    assert widget.region_table_model.rowCount() == 2
    assert widget.region_table_model.columnCount() == 2


def test_table_model_header_labels(regions_widget_with_layer):
    """Test that table model header labels are correct."""
    widget, _ = regions_widget_with_layer
    assert widget.region_table_model.headerData(0, Qt.Horizontal) == "Name"
    assert (
        widget.region_table_model.headerData(1, Qt.Horizontal) == "Shape type"
    )


@pytest.mark.parametrize(
    "column_index",
    [
        pytest.param(0, id="name_column"),
        pytest.param(1, id="shape_type_column"),
    ],
)
@pytest.mark.parametrize(
    "role",
    [
        pytest.param(Qt.DisplayRole, id="display_role"),
        pytest.param(Qt.EditRole, id="edit_role"),
    ],
)
def test_table_model_data_returns_correct_values(
    regions_widget_with_layer, column_index, role
):
    """Test that table model returns correct data for each column and role.

    The role captures the reason why Qt calls the .data() method of the
    model. Qt.DisplayRole means Qt wants the data for display.
    Qt.EditRole means Qt wants the data to pre-fill an editor.
    """
    expected = {
        0: {
            Qt.DisplayRole: DEFAULT_REGION_NAME,
            Qt.EditRole: DEFAULT_REGION_NAME,
        },
        1: {Qt.DisplayRole: "polygon", Qt.EditRole: None},
    }
    widget, _ = regions_widget_with_layer
    index = widget.region_table_model.index(0, column_index)
    result = widget.region_table_model.data(index, role)
    assert result == expected[column_index][role]


@pytest.mark.parametrize(
    "method, args, expected",
    [
        pytest.param("data", (Qt.DisplayRole,), None, id="data_returns_none"),
        pytest.param(
            "setData", ("Name", Qt.EditRole), False, id="setData_returns_false"
        ),
    ],
)
def test_table_model_with_stale_index(
    regions_widget_with_layer, method, args, expected
):
    """Test table model methods return appropriate values for a stale index.

    A stale index is one that was valid when created but whose row
    exceeds the layer data after shapes are removed.
    """
    widget, layer = regions_widget_with_layer
    assert len(layer.data) == 2  # row 1 is valid before clearing
    index = widget.region_table_model.index(1, 0)
    layer.data = []  # makes index stale
    result = getattr(widget.region_table_model, method)(index, *args)
    assert result == expected


def test_table_model_setData_updates_region_name(regions_widget_with_layer):
    """Test that setData updates the region name column.

    The `name` column is column index = 0.
    """
    widget, layer = regions_widget_with_layer
    index = widget.region_table_model.index(0, 0)

    # Assert that name is initially default
    assert layer.properties["name"][0] == DEFAULT_REGION_NAME

    # Edit the name via the model and assert it updates in layer properties
    result = widget.region_table_model.setData(index, "New Name", Qt.EditRole)
    assert result is True
    assert layer.properties["name"][0] == "New Name"


def test_table_model_setData_rejects_shape_type_edit(
    regions_widget_with_layer,
):
    """Test that setData returns False for shape_type column.

    The `shape_type` column is column index = 1 .
    """
    widget, _ = regions_widget_with_layer
    index = widget.region_table_model.index(0, 1)
    result = widget.region_table_model.setData(index, "rectangle", Qt.EditRole)
    assert result is False


@pytest.mark.parametrize(
    "column, expected_editable",
    [(0, True), (1, False)],
    ids=["name_editable", "shape_type_not_editable"],
)
def test_table_model_column_editability(
    regions_widget_with_layer, column, expected_editable
):
    """Test that only the name column is editable."""
    widget, _ = regions_widget_with_layer
    index = widget.region_table_model.index(0, column)
    flags = widget.region_table_model.flags(index)
    assert bool(flags & Qt.ItemIsEditable) == expected_editable


def test_table_model_updates_on_shape_added(
    regions_widget_with_layer, two_polygons
):
    """Test that adding a shape updates the table model."""
    widget, layer = regions_widget_with_layer
    initial_count = widget.region_table_model.rowCount()
    layer.add(two_polygons[:1])

    assert widget.region_table_model.rowCount() == initial_count + 1


def test_sync_names_assigns_default_to_new_shapes(
    regions_widget_with_layer, two_polygons
):
    """Test that _sync_names_on_shape_change assigns a unique default name."""
    widget, layer = regions_widget_with_layer
    model = widget.region_table_model
    # Add a shape so layer has 3 shapes
    layer.add(two_polygons[:1])
    # Reset _last_shape_count to simulate state before shape was added
    model._last_shape_count = 2
    # Call sync with assign_default_to_new=True
    model._sync_names_on_shape_change(n_shapes=3, assign_default_to_new=True)
    # New shape gets a unique name derived from DEFAULT_REGION_NAME
    names = layer.properties["name"]
    assert names[2].startswith(DEFAULT_REGION_NAME)
    assert names[2] not in names[:2]  # new name is unique among prior names


def test_table_model_updates_on_shape_removed(regions_widget_with_layer):
    """Test that removing a shape updates the table model."""
    widget, layer = regions_widget_with_layer
    initial_count = widget.region_table_model.rowCount()
    layer.selected_data = {0}
    layer.remove_selected()

    assert widget.region_table_model.rowCount() == initial_count - 1


def test_table_model_set_data_handler_uniquifies_pasted_names(
    regions_widget_with_layer, two_polygons
):
    """Test that _on_layer_set_data uniquifies duplicate names from copy-paste.

    This handler is triggered by copy-paste operations. It should detect
    shape count changes and give pasted shapes unique names.
    """
    widget, layer = regions_widget_with_layer
    model = widget.region_table_model

    # Add a third shape with a name that duplicates an existing one
    layer.add(two_polygons[:1])
    layer.properties = {"name": ["Region-A", "Region-B", "Region-A"]}

    # Simulate the state before a "paste" by resetting the shape count tracker
    model._last_shape_count = 2

    # Call the handler (as would happen on set_data event)
    model._on_layer_set_data()

    # Verify model updated and duplicate name was made unique
    assert model.rowCount() == 3
    assert model._last_shape_count == 3
    expected_names = ["Region-A", "Region-B", "Region-A [1]"]
    assert list(layer.properties["name"]) == expected_names


def test_table_model_cleared_on_layer_deletion(regions_widget_with_layer):
    """Test that deleting the layer clears the table model."""
    widget, layer = regions_widget_with_layer
    widget.viewer.layers.remove(layer)
    assert widget.region_table_model is None


def test_table_model_ignores_other_layer_deletion(
    regions_widget_with_layer, two_polygons
):
    """Test that table model ignores deletion of unrelated layers."""
    widget, layer = regions_widget_with_layer
    other_layer = widget.viewer.add_shapes(two_polygons, name="Other layer")
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


def test_on_layer_deleted_cleans_up_table_model(
    regions_widget_with_layer, mocker
):
    """Test that _on_layer_deleted disconnects and clears the table model."""
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
def test_table_row_selection_syncs_to_shape(regions_widget_with_layer):
    """Test that selecting a row in table selects shape in layer."""
    widget, layer = regions_widget_with_layer
    widget.region_table_view.selectRow(0)
    assert layer.selected_data == {0}


def test_shape_selection_syncs_to_table_row(regions_widget_with_layer):
    """Test that selecting a shape in napari highlights the table row.
    This is the inverse of the previous test, ensuring a bidirectional sync.
    """
    widget, layer = regions_widget_with_layer
    layer.selected_data = {1}
    assert widget.region_table_view.currentIndex().row() == 1


def test_shape_deselection_clears_table_selection(regions_widget_with_layer):
    """Test that deselecting shapes in napari clears the table selection."""
    widget, layer = regions_widget_with_layer
    layer.selected_data = {0}
    layer.selected_data = set()
    assert not widget.region_table_view.selectionModel().hasSelection()


def test_napari_layer_selection_syncs_to_dropdown(regions_widget):
    """Test that selecting a region layer in napari updates the dropdown."""
    viewer = regions_widget.viewer
    layer_a = add_regions_layer(viewer, name="Regions-A")
    layer_b = add_regions_layer(viewer, name="Regions-B")

    viewer.layers.selection.active = layer_a
    assert regions_widget.layer_dropdown.currentText() == "Regions-A"

    viewer.layers.selection.active = layer_b
    assert regions_widget.layer_dropdown.currentText() == "Regions-B"


def test_non_region_layer_selection_does_not_change_dropdown(regions_widget):
    """Test that selecting a non-region layer leaves the dropdown unchanged."""
    viewer = regions_widget.viewer
    add_regions_layer(viewer, name="Regions-A")
    other_layer = viewer.add_shapes(name="Other shapes")

    regions_widget.layer_dropdown.setCurrentText("Regions-A")
    viewer.layers.selection.active = other_layer

    assert regions_widget.layer_dropdown.currentText() == "Regions-A"


def test_table_allows_name_editing(regions_widget_with_layer):
    """Test that name column is editable via double-click."""
    widget, _ = regions_widget_with_layer
    triggers = widget.region_table_view.editTriggers()
    assert triggers & QTableView.DoubleClicked


# ------------------- Tests for tooltips -------------------------------------#
@pytest.mark.parametrize(
    "add_shapes_kwargs, expected_text",
    [
        pytest.param(None, "No region layers", id="no_layers"),
        pytest.param(
            {"name": "Regions"},
            "No regions in this layer",
            id="empty_layer",
        ),
        pytest.param(
            {
                "name": "Regions",
                "data": [[[0, 0], [0, 10], [10, 10], [10, 0]]],
            },
            "Click a row",
            id="with_shapes",
        ),
    ],
)
def test_table_tooltip_reflects_state(
    make_napari_viewer_proxy, add_shapes_kwargs, expected_text
):
    """Test table tooltip text reflects current widget state."""
    viewer = make_napari_viewer_proxy()
    if add_shapes_kwargs is not None:
        add_regions_layer(viewer, **add_shapes_kwargs)
    widget = RegionsWidget(viewer)
    assert expected_text in widget.region_table_view.toolTip()


# ------------------- Tests for edge cases -----------------------------------#
def test_empty_shapes_layer(make_napari_viewer_proxy):
    """Test table handles empty shapes layer."""
    viewer = make_napari_viewer_proxy()
    add_regions_layer(viewer)
    with does_not_raise():
        widget = RegionsWidget(viewer)

    assert widget.region_table_model.rowCount() == 0


def test_table_model_flags_invalid_index(regions_widget_with_layer):
    """Test flags returns NoItemFlags for invalid index."""
    widget, _ = regions_widget_with_layer
    invalid_index = QModelIndex()
    flags = widget.region_table_model.flags(invalid_index)
    assert flags == Qt.NoItemFlags


@pytest.mark.parametrize(
    "call_handler",
    [
        pytest.param(
            lambda v: v._on_row_selection_changed(None, None),
            id="row_selection",
        ),
        pytest.param(
            lambda v: v._on_shape_selection_changed(),
            id="shape_selection",
        ),
    ],
)
def test_table_view_selection_handlers_return_early_with_no_model(
    regions_widget, call_handler
):
    """Test selection handlers do not raise when no table model is linked."""
    with does_not_raise():
        call_handler(regions_widget.region_table_view)


def test_table_view_selection_with_empty_indexes(regions_widget_with_layer):
    """Test table view handles empty selection indexes."""
    widget, _ = regions_widget_with_layer
    empty_selection = QItemSelection()
    with does_not_raise():
        widget.region_table_view._on_row_selection_changed(
            empty_selection, None
        )


@pytest.mark.parametrize(
    "orientation, role, expected",
    [
        (Qt.Vertical, Qt.DisplayRole, "0"),
        (Qt.Horizontal, Qt.DecorationRole, None),
    ],
    ids=["vertical_header", "non_display_role"],
)
def test_table_model_header_data_edge_cases(
    regions_widget_with_layer, orientation, role, expected
):
    """Test headerData for vertical orientation and non-display roles."""
    widget, _ = regions_widget_with_layer
    header = widget.region_table_model.headerData(0, orientation, role)
    assert header == expected
