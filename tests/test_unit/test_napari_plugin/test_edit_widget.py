"""Test the napari plugin edit widget."""

from unittest.mock import Mock

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from napari.utils.theme import get_theme

from movement.napari.edit_widget import (
    AXES_MARGIN_PIXELS,
    LANE_HEIGHT_PIXELS,
    MIN_CANVAS_HEIGHT_PIXELS,
    MIN_VISIBLE_FRAMES,
    EditWidget,
)


def _bar_colors(edit_widget):
    """Return the RGBA colour of each drawn bar, as a list of tuples."""
    return [tuple(bar.get_colors()[0]) for bar in edit_widget._bars]


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


def test_selecting_non_points_layer_keeps_timeline(loader_with_edited_point):
    """Selecting an unrelated layer must not blank the timeline.

    Selecting e.g. an image layer previously reset ``active_layer`` to
    ``None`` and collapsed the frame axis to 0-1; the timeline should
    instead keep showing the last movement Points layer.
    """
    viewer = loader_with_edited_point.viewer
    edit_widget = EditWidget(viewer)
    points_layer = edit_widget.active_layer
    full_xlim = edit_widget.ax.get_xlim()

    other_layer = viewer.add_image(np.zeros((4, 4)))
    viewer.layers.selection.active = other_layer

    assert edit_widget.active_layer is points_layer
    assert edit_widget.ax.get_xlim() == full_xlim


def test_playhead_and_bars_follow_the_napari_theme(loader_with_edited_point):
    """The playhead and edit-bar colours are taken from the napari theme.

    The playhead is the theme's ``secondary`` colour and collapsed bars
    the theme's ``current`` colour; switching the viewer theme re-styles
    both live (``_apply_theme`` recreates the bars).
    """
    viewer = loader_with_edited_point.viewer
    viewer.theme = "dark"
    edit_widget = EditWidget(viewer)

    dark = get_theme("dark")
    assert to_rgba(edit_widget.playhead.get_color()) == to_rgba(
        dark.secondary.as_hex()
    )
    assert edit_widget._edit_bar_color == dark.current.as_hex()
    assert _bar_colors(edit_widget) == [to_rgba(dark.current.as_hex())]

    viewer.theme = "light"
    light = get_theme("light")
    assert to_rgba(edit_widget.playhead.get_color()) == to_rgba(
        light.secondary.as_hex()
    )
    assert edit_widget._edit_bar_color == light.current.as_hex()
    assert _bar_colors(edit_widget) == [to_rgba(light.current.as_hex())]


def test_lanes_collapse_by_frame_or_split_by_individual(
    loader_with_two_edited_individuals,
):
    """Lane structure follows the "Display individuals" toggle.

    Collapsed (the default): one bar per edited *frame* (individuals
    sharing a frame merge into one bar), no y-ticks and no lane
    dividers. Displaying individuals: one bar per (frame, individual),
    one y-tick per individual, and a divider between each lane pair.
    Toggling back collapses everything again.
    """
    edit_widget = EditWidget(loader_with_two_edited_individuals.viewer)
    n_individuals = len(set(edit_widget.active_layer.properties["individual"]))

    # Edits are on frames {2, 5}; frame 2 is shared by both individuals.
    assert len(edit_widget._bars) == 2
    assert list(edit_widget.ax.get_yticks()) == []
    assert edit_widget._lane_dividers == []

    edit_widget.set_show_individuals(True)

    # (2, id_0), (2, id_1) and (5, id_1) -> three separate bars.
    assert len(edit_widget._bars) == 3
    assert len(edit_widget.ax.get_yticks()) == n_individuals
    assert len(edit_widget._lane_dividers) == n_individuals - 1

    edit_widget.set_show_individuals(False)

    assert len(edit_widget._bars) == 2
    assert edit_widget._lane_dividers == []


def test_bar_colours_follow_display_mode_not_edited_data(
    loader_with_two_edited_individuals,
):
    """Bar colours depend on the display mode, not on what was edited.

    Collapsed: every bar is the napari theme's edit-bar colour,
    whatever mix of individuals or keypoints was edited. Displaying
    individuals: each bar takes its individual's face colour, read
    straight from the Points layer. Toggling the option off restores
    the single colour.
    """
    edit_widget = EditWidget(loader_with_two_edited_individuals.viewer)
    single = to_rgba(edit_widget._edit_bar_color)

    assert _bar_colors(edit_widget) == [single, single]

    edit_widget.set_show_individuals(True)
    layer = edit_widget.active_layer
    palette: dict = {}
    for ind, color in zip(
        layer.properties["individual"], layer.face_color, strict=False
    ):
        palette.setdefault(ind, tuple(color))
    # One bar for id_0 (frame 2) and two for id_1 (frames 2 and 5);
    # sorted so the assertion doesn't depend on bar draw order.
    assert sorted(_bar_colors(edit_widget)) == pytest.approx(
        sorted([palette["id_0"], palette["id_1"], palette["id_1"]])
    )

    edit_widget.set_show_individuals(False)
    assert _bar_colors(edit_widget) == [single, single]


def test_canvas_wrapped_in_vertical_scroll_area(loader_with_edited_point):
    """The canvas lives in a width-tracking, vertically scrolling area.

    The scroll area keeps the docked timeline at its usual height even
    when the canvas inside it grows for many individuals, so the user
    never has to resize the dock.
    """
    edit_widget = EditWidget(loader_with_edited_point.viewer)

    assert edit_widget.scroll_area.widget() is edit_widget.canvas
    assert edit_widget.scroll_area.widgetResizable()
    assert edit_widget.scroll_area.minimumHeight() == MIN_CANVAS_HEIGHT_PIXELS

    # Growing the canvas for many lanes must not grow the scroll area.
    edit_widget._show_individuals = True
    edit_widget._fit_canvas_height(40)
    assert edit_widget.scroll_area.minimumHeight() == MIN_CANVAS_HEIGHT_PIXELS


@pytest.mark.parametrize(
    "show_individuals, n_lanes, expected",
    [
        pytest.param(False, 40, MIN_CANVAS_HEIGHT_PIXELS, id="collapsed"),
        pytest.param(True, 2, MIN_CANVAS_HEIGHT_PIXELS, id="few_lanes"),
        pytest.param(
            True,
            40,
            40 * LANE_HEIGHT_PIXELS + AXES_MARGIN_PIXELS,
            id="many_lanes_grow",
        ),
    ],
)
def test_fit_canvas_height_scales_with_lane_count(
    loader_with_edited_point, show_individuals, n_lanes, expected
):
    """Many individual lanes grow the canvas; collapsed/few keep it at min.

    A taller-than-dock canvas is what makes the enclosing scroll area
    show a vertical scrollbar.
    """
    edit_widget = EditWidget(loader_with_edited_point.viewer)
    edit_widget._show_individuals = show_individuals

    edit_widget._fit_canvas_height(n_lanes)

    assert edit_widget.canvas.minimumHeight() == expected
