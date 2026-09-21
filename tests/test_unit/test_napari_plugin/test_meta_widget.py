"""Test the napari plugin meta widget."""

import numpy as np
import pytest
from napari.layers.base import ActionType

from movement.napari.meta_widget import MovementMetaWidget


def test_meta_widget_instantiation(make_napari_viewer_proxy):
    """Test that the meta widget can be properly instantiated."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)

    # number of collapsible widgets
    assert len(meta_widget.collapsible_widgets) == 4
    assert meta_widget.edit_timeline_widget is None

    first_widget = meta_widget.collapsible_widgets[0]
    assert first_widget._text == "Load tracked data"
    assert first_widget.isExpanded()

    second_widget = meta_widget.collapsible_widgets[1]
    assert second_widget._text == "Edit tracked data"
    assert not second_widget.isExpanded()

    third_widget = meta_widget.collapsible_widgets[2]
    assert third_widget._text == "Save tracked data"
    assert not third_widget.isExpanded()

    fourth_widget = meta_widget.collapsible_widgets[3]
    assert fourth_widget._text == "Define regions of interest"
    assert not fourth_widget.isExpanded()


def test_edit_timeline_widget_collapsable_roundtrip(
    make_napari_viewer_proxy,
):
    """Expand, collapse, then re-expand the "Edit tracked data" section."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[1]

    edit_timeline_collapsible.expand(animate=False)
    assert meta_widget.edit_timeline_widget is not None
    assert not meta_widget._edit_timeline_dock_widget.isHidden()

    edit_timeline_collapsible.collapse(animate=False)
    assert (
        meta_widget.edit_timeline_widget is not None
    )  # not torn down, just hidden
    assert meta_widget._edit_timeline_dock_widget.isHidden()

    edit_timeline_collapsible.expand(animate=False)
    assert not meta_widget._edit_timeline_dock_widget.isHidden()


def test_closing_edit_timeline_dock_via_its_x_resets_state(
    make_napari_viewer_proxy,
):
    """Closing the docked timeline via its title-bar "X" resets state.

    ``_on_edit_timeline_dock_gone`` is connected to the dock widget's
    ``destroyed`` signal (see ``MovementMetaWidget.__init__``), which
    fires when napari tears the dock down after the user closes it that
    way -- unlike collapsing the "Edit tracked data" section, which
    only hides it (see ``test_edit_timeline_widget_collapsable_roundtrip``).
    """
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[1]
    edit_timeline_collapsible.expand(animate=False)
    assert meta_widget.edit_timeline_widget is not None

    meta_widget._on_edit_timeline_dock_gone()

    assert meta_widget.edit_timeline_widget is None
    assert meta_widget._edit_timeline_dock_widget is None
    assert not edit_timeline_collapsible.isExpanded()


def test_show_individuals_checkbox_edit_timeline_widget(
    make_napari_viewer_proxy,
):
    """The sidebar checkbox controls the docked timeline's lane display."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[1]
    edit_timeline_collapsible.expand(animate=False)

    meta_widget.edit_controls.show_individuals_checkbox.setChecked(True)
    assert meta_widget.edit_timeline_widget._show_individuals is True

    meta_widget.edit_controls.show_individuals_checkbox.setChecked(False)
    assert meta_widget.edit_timeline_widget._show_individuals is False


@pytest.mark.parametrize(
    "individuals, expect_enabled",
    [
        pytest.param(["id_0"], False, id="single_individual"),
        pytest.param(["id_0", "id_1"], True, id="multiple_individuals"),
    ],
)
def test_show_individuals_checkbox_enabled_only_for_multiple(
    make_napari_viewer_proxy, add_movement_points, individuals, expect_enabled
):
    """Disable "Display individuals" for single-individual datasets."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    checkbox = meta_widget.edit_controls.show_individuals_checkbox

    assert not checkbox.isEnabled()  # nothing loaded yet

    layer = add_movement_points(viewer, individuals)
    viewer.layers.selection.active = layer

    assert checkbox.isEnabled() is expect_enabled


def test_show_individuals_enabled_noop_without_a_movement_layer(
    make_napari_viewer_proxy, add_movement_points
):
    """Selecting a non-movement layer leaves the checkbox state untouched.

    ``_show_individuals_enabled`` fires on every active-layer change,
    but with no movement Points layer left to check for individuals it
    has nothing to enable/disable for and must return early -- without
    that guard it would crash reading ``.properties`` off ``None``.
    """
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    checkbox = meta_widget.edit_controls.show_individuals_checkbox

    multi = add_movement_points(viewer, ["id_0", "id_1"])
    viewer.layers.selection.active = multi
    assert checkbox.isEnabled()  # sanity: enabled for multi-individual data

    viewer.layers.remove(multi)
    other = viewer.add_points(np.zeros((1, 2)))  # not a movement layer
    viewer.layers.selection.active = other

    assert checkbox.isEnabled()  # unchanged: no movement layer to check


def test_show_individuals_unchecked_when_switching_to_single(
    make_napari_viewer_proxy, add_movement_points
):
    """Switching to a single-individual layer clears an active check."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    checkbox = meta_widget.edit_controls.show_individuals_checkbox

    multi = add_movement_points(viewer, ["id_0", "id_1"])
    viewer.layers.selection.active = multi
    checkbox.setChecked(True)

    single = add_movement_points(viewer)
    viewer.layers.selection.active = single

    assert not checkbox.isEnabled()
    assert not checkbox.isChecked()


def test_expanding_edit_section_autoselects_points_layer(
    make_napari_viewer_proxy, add_movement_points
):
    """Expanding the section makes a movement Points layer active."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[1]

    points_layer = add_movement_points(viewer)
    # An unrelated layer stealing the active selection.
    other_layer = viewer.add_points(np.zeros((1, 2)))
    viewer.layers.selection.active = other_layer

    edit_timeline_collapsible.expand(animate=False)

    assert viewer.layers.selection.active.name == points_layer.name
    assert (
        meta_widget.edit_timeline_widget.active_layer.name == points_layer.name
    )


def test_expanding_edit_section_keeps_movement_layer_active(
    make_napari_viewer_proxy, add_movement_points
):
    """A movement Points layer already active is left selected."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[1]

    add_movement_points(viewer)
    second_layer = add_movement_points(viewer)
    viewer.layers.selection.active = second_layer

    edit_timeline_collapsible.expand(animate=False)

    assert viewer.layers.selection.active.name == second_layer.name


@pytest.mark.parametrize(
    "edited, pre_expanded",
    [
        pytest.param(False, True, id="no_edits_forces_collapse"),
        pytest.param(True, True, id="prior_edits_still_collapse"),
    ],
)
def test_edit_section_stays_collapsed_on_load(
    make_napari_viewer_proxy, add_movement_points, edited, pre_expanded
):
    """Loading a layer always collapses the edit timeline section.

    Prior edits in the dataset no longer auto-open the section; it
    only opens once a point is edited in this session. A manual open
    is overridden on load.
    """
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[1]
    if pre_expanded:
        edit_timeline_collapsible.expand(
            animate=False
        )  # simulate a manual open
    else:
        assert (
            not edit_timeline_collapsible.isExpanded()
        )  # collapsed by default

    add_movement_points(viewer, edited=[edited])

    assert not edit_timeline_collapsible.isExpanded()


@pytest.mark.parametrize(
    "action, expect_expanded",
    [
        pytest.param(ActionType.CHANGED, True, id="drag_expands"),
        pytest.param(ActionType.REMOVING, True, id="remove_expands"),
        pytest.param(ActionType.ADDED, False, id="add_does_not_expand"),
    ],
)
def test_editing_points_expands_edit_section(
    make_napari_viewer_proxy,
    add_movement_points,
    mocker,
    action,
    expect_expanded,
):
    """Dragging or removing a point opens the "Edit tracked data" section."""
    # ``_on_points_edited`` defers the expand via ``QTimer.singleShot``;
    # run the callback synchronously so the test does not pump the loop.
    mocker.patch(
        "movement.napari.meta_widget.QTimer.singleShot",
        side_effect=lambda _ms, cb: cb(),
    )
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_timeline_collapsible = meta_widget.collapsible_widgets[1]

    layer = add_movement_points(viewer)
    assert not edit_timeline_collapsible.isExpanded()

    layer.events.data(
        value=layer.data,
        action=action,
        data_indices=(0,),
        vertex_indices=((),),
    )

    assert edit_timeline_collapsible.isExpanded() is expect_expanded
