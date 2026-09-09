"""Test the napari plugin meta widget."""

import numpy as np
import pytest
from napari.layers.base import ActionType

from movement.napari.loader_widgets import POINTS_LAYER_KEY
from movement.napari.meta_widget import MovementMetaWidget


def test_meta_widget_instantiation(make_napari_viewer_proxy):
    """Test that the meta widget can be properly instantiated."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)

    # number of collapsible widgets
    assert len(meta_widget.collapsible_widgets) == 4
    assert meta_widget.edit_widget is None

    first_widget = meta_widget.collapsible_widgets[0]
    assert first_widget._text == "Load tracked data"
    assert first_widget.isExpanded()

    second_widget = meta_widget.collapsible_widgets[1]
    assert second_widget._text == "Define regions of interest"
    assert not second_widget.isExpanded()

    third_widget = meta_widget.collapsible_widgets[2]
    assert third_widget._text == "Edit tracked data"
    assert not third_widget.isExpanded()

    fourth_widget = meta_widget.collapsible_widgets[3]
    assert fourth_widget._text == "Save tracked data"
    assert not fourth_widget.isExpanded()


def test_edit_widget_collapsable_roundtrip(
    make_napari_viewer_proxy,
):
    """Expand, collapse, then re-expand the "Edit tracked data" section."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_collapsible = meta_widget.collapsible_widgets[2]

    edit_collapsible.expand(animate=False)
    assert meta_widget.edit_widget is not None
    assert not meta_widget._edit_dock_widget.isHidden()

    edit_collapsible.collapse(animate=False)
    assert meta_widget.edit_widget is not None  # not torn down, just hidden
    assert meta_widget._edit_dock_widget.isHidden()

    edit_collapsible.expand(animate=False)
    assert not meta_widget._edit_dock_widget.isHidden()


def test_show_individuals_checkbox_edit_widget(
    make_napari_viewer_proxy,
):
    """The sidebar checkbox controls the docked timeline's lane display."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_collapsible = meta_widget.collapsible_widgets[2]
    edit_collapsible.expand(animate=False)

    meta_widget.edit_controls.show_individuals_checkbox.setChecked(True)
    assert meta_widget.edit_widget._show_individuals is True

    meta_widget.edit_controls.show_individuals_checkbox.setChecked(False)
    assert meta_widget.edit_widget._show_individuals is False


@pytest.mark.parametrize(
    "individuals, expect_enabled",
    [
        pytest.param(["id_0"], False, id="single_individual"),
        pytest.param(["id_0", "id_1"], True, id="multiple_individuals"),
    ],
)
def test_show_individuals_checkbox_enabled_only_for_multiple(
    make_napari_viewer_proxy, individuals, expect_enabled
):
    """Disable "Display individuals" for single-individual datasets."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    checkbox = meta_widget.edit_controls.show_individuals_checkbox

    assert not checkbox.isEnabled()  # nothing loaded yet

    n = len(individuals)
    layer = viewer.add_points(
        np.zeros((n, 2)),
        properties={
            "edited": np.array([False] * n),
            "individual": np.array(individuals),
        },
        metadata={POINTS_LAYER_KEY: True},
    )
    viewer.layers.selection.active = layer

    assert checkbox.isEnabled() is expect_enabled


def test_show_individuals_unchecked_when_switching_to_single(
    make_napari_viewer_proxy,
):
    """Switching to a single-individual layer clears an active check."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    checkbox = meta_widget.edit_controls.show_individuals_checkbox

    multi = viewer.add_points(
        np.zeros((2, 2)),
        properties={
            "edited": np.array([False, False]),
            "individual": np.array(["id_0", "id_1"]),
        },
        metadata={POINTS_LAYER_KEY: True},
    )
    viewer.layers.selection.active = multi
    checkbox.setChecked(True)

    single = viewer.add_points(
        np.zeros((1, 2)),
        properties={
            "edited": np.array([False]),
            "individual": np.array(["id_0"]),
        },
        metadata={POINTS_LAYER_KEY: True},
    )
    viewer.layers.selection.active = single

    assert not checkbox.isEnabled()
    assert not checkbox.isChecked()


def test_expanding_edit_section_autoselects_points_layer(
    make_napari_viewer_proxy,
):
    """Expanding the section makes a movement Points layer active."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_collapsible = meta_widget.collapsible_widgets[2]

    points_layer = viewer.add_points(
        np.zeros((1, 2)),
        properties={
            "edited": np.array([False]),
            "individual": np.array(["id_0"]),
        },
        metadata={POINTS_LAYER_KEY: True},
    )
    # An unrelated layer stealing the active selection.
    other_layer = viewer.add_points(np.zeros((1, 2)))
    viewer.layers.selection.active = other_layer

    edit_collapsible.expand(animate=False)

    assert viewer.layers.selection.active.name == points_layer.name
    assert meta_widget.edit_widget.active_layer.name == points_layer.name


def test_expanding_edit_section_keeps_movement_layer_active(
    make_napari_viewer_proxy,
):
    """A movement Points layer already active is left selected."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_collapsible = meta_widget.collapsible_widgets[2]

    viewer.add_points(
        np.zeros((1, 2)),
        properties={
            "edited": np.array([False]),
            "individual": np.array(["id_0"]),
        },
        metadata={POINTS_LAYER_KEY: True},
    )
    second_layer = viewer.add_points(
        np.zeros((1, 2)),
        properties={
            "edited": np.array([False]),
            "individual": np.array(["id_0"]),
        },
        metadata={POINTS_LAYER_KEY: True},
    )
    viewer.layers.selection.active = second_layer

    edit_collapsible.expand(animate=False)

    assert viewer.layers.selection.active.name == second_layer.name


@pytest.mark.parametrize(
    "edited, pre_expanded",
    [
        pytest.param(False, True, id="no_edits_forces_collapse"),
        pytest.param(True, True, id="prior_edits_still_collapse"),
    ],
)
def test_edit_section_stays_collapsed_on_load(
    make_napari_viewer_proxy, edited, pre_expanded
):
    """Loading a layer always collapses the edit section.

    Prior edits in the dataset no longer auto-open the section; it
    only opens once a point is edited in this session. A manual open
    is overridden on load.
    """
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    edit_collapsible = meta_widget.collapsible_widgets[2]
    if pre_expanded:
        edit_collapsible.expand(animate=False)  # simulate a manual open
    else:
        assert not edit_collapsible.isExpanded()  # collapsed by default

    viewer.add_points(
        np.zeros((1, 2)),
        properties={
            "edited": np.array([edited]),
            "individual": np.array(["id_0"]),
        },
        metadata={POINTS_LAYER_KEY: True},
    )

    assert not edit_collapsible.isExpanded()


@pytest.mark.parametrize(
    "action, expect_expanded",
    [
        pytest.param(ActionType.CHANGED, True, id="drag_expands"),
        pytest.param(ActionType.REMOVING, True, id="remove_expands"),
        pytest.param(ActionType.ADDED, False, id="add_does_not_expand"),
    ],
)
def test_editing_points_expands_edit_section(
    make_napari_viewer_proxy, mocker, action, expect_expanded
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
    edit_collapsible = meta_widget.collapsible_widgets[2]

    layer = viewer.add_points(
        np.zeros((1, 2)),
        properties={
            "edited": np.array([False]),
            "individual": np.array(["id_0"]),
        },
        metadata={POINTS_LAYER_KEY: True},
    )
    assert not edit_collapsible.isExpanded()

    layer.events.data(
        value=layer.data,
        action=action,
        data_indices=(0,),
        vertex_indices=((),),
    )

    assert edit_collapsible.isExpanded() is expect_expanded
