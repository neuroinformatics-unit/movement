"""Test the napari plugin edit widget."""

from unittest.mock import Mock

import pytest

from movement.napari.edit_widget import MIN_VISIBLE_FRAMES, EditWidget


@pytest.mark.parametrize(
    "click_offset, expect_jump",
    [
        pytest.param(0.0, True, id="exact_bar"),
        pytest.param(0.9, True, id="within_tolerance_of_bar"),
        pytest.param(5.0, False, id="outside_tolerance"),
    ],
)
def test_click_on_timeline_jumps_only_within_tolerance(
    loader_with_edited_point, click_on_timeline, click_offset, expect_jump
):
    """Clicking near a flagged bar jumps the viewer there.

    A bar is a single vertical line, so a click is rarely pixel-exact;
    ``_handle_click`` accepts anything within ``CLICK_TOLERANCE_FRACTION``
    of the bar as a hit on it, and ignores clicks further away.
    """
    viewer = loader_with_edited_point.viewer
    edit_widget = EditWidget(viewer)
    edited_frame = 2

    viewer.dims.current_step = (0,) + viewer.dims.current_step[1:]
    click_on_timeline(edit_widget, xdata=edited_frame + click_offset)

    expected_frame = edited_frame if expect_jump else 0
    assert viewer.dims.current_step[0] == expected_frame


def test_scroll_up_zooms_in_and_down_zooms_out(loader_with_edited_point):
    """Scrolling up shrinks the visible frame range; down grows it."""
    viewer = loader_with_edited_point.viewer
    edit_widget = EditWidget(viewer)
    xmin, xmax = edit_widget.ax.get_xlim()
    cursor = (xmin + xmax) / 2

    edit_widget._on_scroll(
        Mock(inaxes=edit_widget.ax, xdata=cursor, button="up")
    )
    zoomed_in_xmin, zoomed_in_xmax = edit_widget.ax.get_xlim()
    zoomed_in_span = zoomed_in_xmax - zoomed_in_xmin
    assert zoomed_in_span < (xmax - xmin)

    edit_widget._on_scroll(
        Mock(inaxes=edit_widget.ax, xdata=cursor, button="down")
    )
    zoomed_out_xmin, zoomed_out_xmax = edit_widget.ax.get_xlim()
    zoomed_out_span = zoomed_out_xmax - zoomed_out_xmin
    assert zoomed_out_span > zoomed_in_span


@pytest.mark.parametrize(
    "button",
    [
        pytest.param("up", id="zoom_in_floor"),
        pytest.param("down", id="zoom_out_ceiling"),
    ],
)
def test_scroll_repeatedly_clamps_at_span_limit(
    loader_with_edited_point, button
):
    """Scrolling repeatedly in one direction clamps at that span's limit.

    Zooms in a few times first, so the "zoom out" case has room to
    actually grow back towards the ceiling -- the timeline opens
    already fully zoomed out (``_reset_xlim`` runs on construction),
    so without this, scrolling down would trivially no-op from the
    start instead of exercising the ceiling clamp.
    """
    viewer = loader_with_edited_point.viewer
    edit_widget = EditWidget(viewer)
    cursor = sum(edit_widget.ax.get_xlim()) / 2

    def scroll(direction):
        edit_widget._on_scroll(
            Mock(inaxes=edit_widget.ax, xdata=cursor, button=direction)
        )

    for _ in range(3):
        scroll("up")

    for _ in range(50):  # far more scrolls than needed to hit the limit
        scroll(button)

    xmin, xmax = edit_widget.ax.get_xlim()
    expected_span = (
        MIN_VISIBLE_FRAMES
        if button == "up"
        else max(edit_widget._max_frame, 1)
    )
    assert xmax - xmin == pytest.approx(expected_span)


@pytest.mark.parametrize(
    "event_kwargs",
    [
        pytest.param({"inaxes": None, "xdata": 5}, id="outside_axes"),
        pytest.param({"xdata": None}, id="no_xdata"),
    ],
)
def test_scroll_outside_axes_or_without_xdata_is_a_noop(
    loader_with_edited_point, event_kwargs
):
    """A scroll event outside the timeline, or with no xdata, is ignored."""
    viewer = loader_with_edited_point.viewer
    edit_widget = EditWidget(viewer)
    event_kwargs.setdefault("inaxes", edit_widget.ax)
    before = edit_widget.ax.get_xlim()

    edit_widget._on_scroll(Mock(button="up", **event_kwargs))

    assert edit_widget.ax.get_xlim() == before


def test_double_click_resets_zoomed_view(
    loader_with_edited_point, click_on_timeline
):
    """Double-clicking the timeline resets it to the full frame range."""
    viewer = loader_with_edited_point.viewer
    edit_widget = EditWidget(viewer)
    full_xlim = edit_widget.ax.get_xlim()
    cursor = sum(full_xlim) / 2

    edit_widget._on_scroll(
        Mock(inaxes=edit_widget.ax, xdata=cursor, button="up")
    )
    assert edit_widget.ax.get_xlim() != full_xlim  # sanity: actually zoomed

    click_on_timeline(edit_widget, dblclick=True)

    assert edit_widget.ax.get_xlim() == full_xlim


def test_redraw_bars_splits_lanes_by_individual(
    valid_poses_path_and_ds, loaded_data_loader, move_point
):
    """``_redraw_bars`` draws one bar per frame, or one per individual.

    Two individuals are edited on the same frame. With lanes collapsed
    (the default), that's a single shared bar. With "Display
    individuals" on, each individual gets its own lane and bar, even
    though they share a frame.
    """
    filepath, ds = valid_poses_path_and_ds
    loader = loaded_data_loader(filepath, ds)
    move_point(
        loader,
        frame=2,
        keypoint="centroid",
        individual="id_0",
        new_y=100,
        new_x=200,
    )
    move_point(
        loader,
        frame=2,
        keypoint="centroid",
        individual="id_1",
        new_y=150,
        new_x=250,
    )
    edit_widget = EditWidget(loader.viewer)

    assert len(edit_widget._bars) == 1
    assert list(edit_widget.ax.get_yticks()) == []

    edit_widget.set_show_individuals(True)

    assert len(edit_widget._bars) == 2
    assert len(edit_widget.ax.get_yticks()) == 2
