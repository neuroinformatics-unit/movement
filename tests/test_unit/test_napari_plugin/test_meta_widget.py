"""Test the napari plugin meta widget."""

import numpy as np
import pytest

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
    assert third_widget._text == "Edited track Data"
    assert not third_widget.isExpanded()

    fourth_widget = meta_widget.collapsible_widgets[3]
    assert fourth_widget._text == "Save tracked data"
    assert not fourth_widget.isExpanded()


def test_edit_widget_collapsable_roundtrip(
    make_napari_viewer_proxy,
):
    """Expand, collapse, then re-expand the "Edited track Data" section."""
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
    "edited, pre_expanded, expect_expanded",
    [
        pytest.param(False, True, False, id="no_edits_forces_collapse"),
        pytest.param(True, False, True, id="prior_edits_forces_expand"),
    ],
)
def test_edit_section_state_reflects_prior_edits(
    make_napari_viewer_proxy, edited, pre_expanded, expect_expanded
):
    """Loading a layer sets the edit section's state from its edits.

    A never-edited dataset collapses the section, even overriding a
    manual open. A dataset with prior edits expands it and lazily
    creates the edit widget.
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

    assert edit_collapsible.isExpanded() is expect_expanded
    if expect_expanded:
        assert meta_widget.edit_widget is not None
