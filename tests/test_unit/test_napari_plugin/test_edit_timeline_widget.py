"""Test the napari plugin edit timeline widget."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.layers.base import ActionType
from napari.utils.theme import get_theme

from movement.napari.edit_timeline_widget import (
    DRAG_THRESHOLD_PIXELS,
    MAX_LANES_WITH_LABELS,
    MIN_VISIBLE_FRAMES,
    EditTimelineWidget,
)
from movement.napari.layer_wiring import POINTS_LAYER_KEY
from movement.napari.loader_widgets import POINTS_PROPERTIES_KEY


def _bar_colors(edit_timeline_widget):
    """Return the RGBA colour of each drawn bar, as a list of tuples."""
    return [tuple(bar.get_colors()[0]) for bar in edit_timeline_widget._bars]


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
    edit_timeline_widget = EditTimelineWidget(viewer)
    edited_frame = 2

    viewer.dims.current_step = (0,) + viewer.dims.current_step[1:]
    click_on_timeline(edit_timeline_widget, xdata=edited_frame + click_offset)

    expected_frame = edited_frame if expect_jump else 0
    assert viewer.dims.current_step[0] == expected_frame


def test_drag_pans_the_timeline(loader_with_edited_point):
    """Dragging the mouse across the timeline pans the visible range."""
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)
    # Zoom in first: fully zoomed out (the default) there's nowhere to
    # pan to, since the full frame range is already in view.
    edit_timeline_widget._on_scroll(
        Mock(
            inaxes=edit_timeline_widget.ax,
            xdata=sum(edit_timeline_widget.ax.get_xlim()) / 2,
            button="up",
        )
    )
    xmin, xmax = edit_timeline_widget.ax.get_xlim()
    cursor = (xmin + xmax) / 2

    edit_timeline_widget._on_mouse_press(
        Mock(inaxes=edit_timeline_widget.ax, xdata=cursor, x=100)
    )
    edit_timeline_widget._on_mouse_motion(
        Mock(x=50)
    )  # dragged left by 50 pixels
    edit_timeline_widget._on_mouse_release(Mock())

    assert edit_timeline_widget.ax.get_xlim() != (xmin, xmax)


def test_mouse_press_outside_axes_starts_no_drag(loader_with_edited_point):
    """A press outside the timeline (or with no xdata) is ignored."""
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)

    edit_timeline_widget._on_mouse_press(Mock(inaxes=None, xdata=1.0, x=100))

    assert edit_timeline_widget._press_pixel_x is None


def test_mouse_motion_without_a_prior_press_is_a_noop(
    loader_with_edited_point,
):
    """Mouse motion before any press on the timeline pans nothing."""
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)
    xlim_before = edit_timeline_widget.ax.get_xlim()

    edit_timeline_widget._on_mouse_motion(Mock(x=50))

    assert edit_timeline_widget.ax.get_xlim() == xlim_before


def test_small_mouse_movement_is_not_treated_as_a_drag(
    loader_with_edited_point,
):
    """Movement below the drag threshold does not pan the timeline."""
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)
    cursor = sum(edit_timeline_widget.ax.get_xlim()) / 2
    edit_timeline_widget._on_mouse_press(
        Mock(inaxes=edit_timeline_widget.ax, xdata=cursor, x=100)
    )
    xlim_before = edit_timeline_widget.ax.get_xlim()

    edit_timeline_widget._on_mouse_motion(Mock(x=100 + DRAG_THRESHOLD_PIXELS))

    assert edit_timeline_widget.ax.get_xlim() == xlim_before
    assert edit_timeline_widget._dragged is False


def test_mouse_release_without_a_prior_press_is_a_noop(
    loader_with_edited_point,
):
    """Releasing without a preceding press on the timeline does nothing."""
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)

    edit_timeline_widget._on_mouse_release(Mock())  # should not raise


def test_release_without_drag_jumps_to_the_clicked_frame(
    loader_with_edited_point,
):
    """A press+release with no real movement in between is a click.

    Exercises the real ``_on_mouse_press``/``_on_mouse_release`` flow,
    unlike the ``click_on_timeline`` fixture, which calls
    ``_handle_click`` directly and so never touches this "was it a
    click or a drag?" bookkeeping.
    """
    viewer = loader_with_edited_point.viewer
    edit_timeline_widget = EditTimelineWidget(viewer)
    edited_frame = 2
    viewer.dims.current_step = (0,) + viewer.dims.current_step[1:]

    edit_timeline_widget._on_mouse_press(
        Mock(
            inaxes=edit_timeline_widget.ax,
            xdata=edited_frame,
            x=100,
            dblclick=False,
        )
    )
    edit_timeline_widget._on_mouse_release(Mock())

    assert viewer.dims.current_step[0] == edited_frame


def test_click_on_timeline_with_no_edited_frames_is_a_noop(
    loader_with_edited_point, click_on_timeline
):
    """Clicking the timeline when nothing is flagged does nothing."""
    viewer = loader_with_edited_point.viewer
    edit_timeline_widget = EditTimelineWidget(viewer)
    edit_timeline_widget._edited_frames = np.array([])
    viewer.dims.current_step = (0,) + viewer.dims.current_step[1:]

    click_on_timeline(edit_timeline_widget, xdata=2)

    assert viewer.dims.current_step[0] == 0


def test_scroll_up_zooms_in_and_down_zooms_out(loader_with_edited_point):
    """Scrolling up shrinks the visible frame range; down grows it."""
    viewer = loader_with_edited_point.viewer
    edit_timeline_widget = EditTimelineWidget(viewer)
    xmin, xmax = edit_timeline_widget.ax.get_xlim()
    cursor = (xmin + xmax) / 2

    edit_timeline_widget._on_scroll(
        Mock(inaxes=edit_timeline_widget.ax, xdata=cursor, button="up")
    )
    zoomed_in_xmin, zoomed_in_xmax = edit_timeline_widget.ax.get_xlim()
    zoomed_in_span = zoomed_in_xmax - zoomed_in_xmin
    assert zoomed_in_span < (xmax - xmin)

    edit_timeline_widget._on_scroll(
        Mock(inaxes=edit_timeline_widget.ax, xdata=cursor, button="down")
    )
    zoomed_out_xmin, zoomed_out_xmax = edit_timeline_widget.ax.get_xlim()
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
    edit_timeline_widget = EditTimelineWidget(viewer)
    cursor = sum(edit_timeline_widget.ax.get_xlim()) / 2

    def scroll(direction):
        edit_timeline_widget._on_scroll(
            Mock(
                inaxes=edit_timeline_widget.ax, xdata=cursor, button=direction
            )
        )

    for _ in range(3):
        scroll("up")

    for _ in range(50):  # far more scrolls than needed to hit the limit
        scroll(button)

    xmin, xmax = edit_timeline_widget.ax.get_xlim()
    expected_span = (
        MIN_VISIBLE_FRAMES
        if button == "up"
        else max(edit_timeline_widget._max_frame, 1)
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
    edit_timeline_widget = EditTimelineWidget(viewer)
    event_kwargs.setdefault("inaxes", edit_timeline_widget.ax)
    before = edit_timeline_widget.ax.get_xlim()

    edit_timeline_widget._on_scroll(Mock(button="up", **event_kwargs))

    assert edit_timeline_widget.ax.get_xlim() == before


def test_double_click_resets_zoomed_view(
    loader_with_edited_point, click_on_timeline
):
    """Double-clicking the timeline resets it to the full frame range."""
    viewer = loader_with_edited_point.viewer
    edit_timeline_widget = EditTimelineWidget(viewer)
    full_xlim = edit_timeline_widget.ax.get_xlim()
    cursor = sum(full_xlim) / 2

    edit_timeline_widget._on_scroll(
        Mock(inaxes=edit_timeline_widget.ax, xdata=cursor, button="up")
    )
    assert (
        edit_timeline_widget.ax.get_xlim() != full_xlim
    )  # sanity: actually zoomed

    click_on_timeline(edit_timeline_widget, dblclick=True)

    assert edit_timeline_widget.ax.get_xlim() == full_xlim


def test_selecting_non_points_layer_keeps_timeline(loader_with_edited_point):
    """Selecting an unrelated layer must not blank the timeline.

    Selecting e.g. an image layer previously reset ``active_layer`` to
    ``None`` and collapsed the frame axis to 0-1; the timeline should
    instead keep showing the last movement Points layer.
    """
    viewer = loader_with_edited_point.viewer
    edit_timeline_widget = EditTimelineWidget(viewer)
    points_layer = edit_timeline_widget.active_layer
    full_xlim = edit_timeline_widget.ax.get_xlim()

    other_layer = viewer.add_image(np.zeros((4, 4)))
    viewer.layers.selection.active = other_layer

    assert edit_timeline_widget.active_layer is points_layer
    assert edit_timeline_widget.ax.get_xlim() == full_xlim


def test_reselecting_the_active_layer_is_a_noop(
    loader_with_edited_point, mocker
):
    """Re-selecting the layer already shown redraws nothing."""
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)
    redraw = mocker.spy(edit_timeline_widget, "_redraw_bars")

    # same active layer as before
    edit_timeline_widget._on_active_layer_changed()

    redraw.assert_not_called()


def test_reconstruct_previously_removed_points_without_a_layer():
    """No layer means nothing to reconstruct."""
    assert (
        EditTimelineWidget._reconstruct_previously_removed_points(None) == []
    )


@pytest.mark.parametrize(
    "times, removed_idx",
    [
        pytest.param([0, 1, 2], 1, id="integer_times_fps1"),
        pytest.param([0.0, 0.5, 1.0], 1, id="subsecond_times_fps2"),
        pytest.param([0.0, 0.1, 0.2, 0.3], 2, id="subsecond_times_fps10"),
        pytest.param([0.0, 0.5, 1.0], 2, id="removed_at_last_frame"),
    ],
)
def test_reconstruct_removed_point_maps_time_to_frame_index(
    times, removed_idx
):
    """A saved removal reconstructs at its frame index, not its time.

    The timeline is indexed by integer frame, but a row's ``time`` is in
    seconds when ``fps`` is set. A point removed and saved from a dataset
    with ``fps != 1`` must map back to its ordinal frame position.
    """
    n = len(times)
    is_removed = [i == removed_idx for i in range(n)]
    properties = pd.DataFrame(
        {
            "time": times,
            "individual": ["id_0"] * n,
            "position_is_nan": is_removed,
            "edited": is_removed,
        }
    )
    layer = Mock(metadata={POINTS_PROPERTIES_KEY: properties})

    result = EditTimelineWidget._reconstruct_previously_removed_points(layer)

    assert result == [(removed_idx, "id_0")]


def test_reconstruct_previously_removed_points_without_any_removed_ones():
    """Edited points that were never removed leave nothing to reconstruct."""
    properties = pd.DataFrame(
        {
            "time": [0, 1],
            "individual": ["id_0", "id_0"],
            "position_is_nan": [False, False],
            "edited": [True, False],
        }
    )
    layer = Mock(metadata={POINTS_PROPERTIES_KEY: properties})

    assert (
        EditTimelineWidget._reconstruct_previously_removed_points(layer) == []
    )


def test_layer_data_changed_ignores_other_layers(
    loader_with_edited_point, mocker
):
    """Data changes on a layer other than the active one are ignored."""
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)
    redraw = mocker.spy(edit_timeline_widget, "_redraw_bars")

    edit_timeline_widget._on_layer_data_changed(Mock(source=object()))

    redraw.assert_not_called()


def test_moving_a_point_defers_a_redraw(loader_with_edited_point, mocker):
    """A live drag on the active layer redraws the bars once Qt catches up.

    The redraw is deferred via ``QTimer.singleShot`` so that
    ``DataLoader``'s own handler (which sets the ``edited`` property
    this widget reads) runs first; run the callback synchronously here
    so the test does not need to pump the Qt event loop.
    """
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)
    mocker.patch(
        "movement.napari.edit_timeline_widget.QTimer.singleShot",
        side_effect=lambda _ms, cb: cb(),
    )
    redraw = mocker.spy(edit_timeline_widget, "_redraw_bars")

    edit_timeline_widget._on_layer_data_changed(
        Mock(
            source=edit_timeline_widget.active_layer, action=ActionType.CHANGED
        )
    )

    redraw.assert_called_once()


def test_removing_a_point_captures_it_and_redraws(loader_with_edited_point):
    """Removing a point on the active layer snapshots it, then redraws.

    The row is about to be deleted from the layer entirely, so its
    identity is captured while the data is still intact (``REMOVING``
    fires before the removal actually happens).
    """
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)
    assert edit_timeline_widget._removed_points == []

    edit_timeline_widget._on_layer_data_changed(
        Mock(
            source=edit_timeline_widget.active_layer,
            action=ActionType.REMOVING,
            data_indices=(0,),
        )
    )

    assert len(edit_timeline_widget._removed_points) == 1


def test_step_changed_leaves_playhead_alone_without_a_current_step(
    loader_with_edited_point,
):
    """No current step (e.g. dims not yet set up) leaves the playhead put."""
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)
    xdata_before = list(edit_timeline_widget.playhead.get_xdata())
    edit_timeline_widget.viewer = Mock(dims=Mock(current_step=()))

    edit_timeline_widget._on_step_changed()

    assert list(edit_timeline_widget.playhead.get_xdata()) == xdata_before


def test_playhead_and_bars_follow_the_napari_theme(loader_with_edited_point):
    """The playhead and edit-bar colours are taken from the napari theme.

    The playhead is the theme's ``secondary`` colour and collapsed bars
    the theme's ``current`` colour; switching the viewer theme re-styles
    both live (``_apply_theme`` recreates the bars).
    """
    viewer = loader_with_edited_point.viewer
    viewer.theme = "dark"
    edit_timeline_widget = EditTimelineWidget(viewer)

    dark = get_theme("dark")
    assert to_rgba(edit_timeline_widget.playhead.get_color()) == to_rgba(
        dark.secondary.as_hex()
    )
    assert edit_timeline_widget._edit_bar_color == dark.current.as_hex()
    assert _bar_colors(edit_timeline_widget) == [
        to_rgba(dark.current.as_hex())
    ]

    viewer.theme = "light"
    light = get_theme("light")
    assert to_rgba(edit_timeline_widget.playhead.get_color()) == to_rgba(
        light.secondary.as_hex()
    )
    assert edit_timeline_widget._edit_bar_color == light.current.as_hex()
    assert _bar_colors(edit_timeline_widget) == [
        to_rgba(light.current.as_hex())
    ]


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
    edit_timeline_widget = EditTimelineWidget(
        loader_with_two_edited_individuals.viewer
    )
    n_individuals = len(
        set(edit_timeline_widget.active_layer.properties["individual"])
    )

    # Edits are on frames {2, 5}; frame 2 is shared by both individuals.
    assert len(edit_timeline_widget._bars) == 2
    assert list(edit_timeline_widget.ax.get_yticks()) == []
    assert edit_timeline_widget._lane_dividers == []

    edit_timeline_widget.set_show_individuals(True)

    # (2, id_0), (2, id_1) and (5, id_1) -> three separate bars.
    assert len(edit_timeline_widget._bars) == 3
    assert len(edit_timeline_widget.ax.get_yticks()) == n_individuals
    assert len(edit_timeline_widget._lane_dividers) == n_individuals - 1

    edit_timeline_widget.set_show_individuals(False)

    assert len(edit_timeline_widget._bars) == 2
    assert edit_timeline_widget._lane_dividers == []


@pytest.mark.parametrize(
    "n_individuals, show_labels",
    [
        pytest.param(MAX_LANES_WITH_LABELS - 1, True, id="below_limit"),
        pytest.param(MAX_LANES_WITH_LABELS, True, id="at_limit"),
        pytest.param(MAX_LANES_WITH_LABELS + 1, False, id="above_limit"),
    ],
)
def test_many_individual_lanes_hide_only_unreadable_axis_elements(
    make_napari_viewer_proxy,
    add_movement_points,
    n_individuals,
    show_labels,
):
    """Keep individual edit bars while hiding crowded axis details."""
    viewer = make_napari_viewer_proxy()
    individuals = [f"id_{index}" for index in range(n_individuals)]
    add_movement_points(viewer, individuals, edited=[True] * n_individuals)
    edit_timeline_widget = EditTimelineWidget(viewer)

    edit_timeline_widget.set_show_individuals(True)

    assert len(edit_timeline_widget._bars) == n_individuals
    if show_labels:
        assert len(edit_timeline_widget.ax.get_yticks()) == n_individuals
        assert len(edit_timeline_widget._lane_dividers) == n_individuals - 1
    else:
        assert list(edit_timeline_widget.ax.get_yticks()) == []
        assert edit_timeline_widget._lane_dividers == []


def test_empty_individual_lanes_do_not_raise(make_napari_viewer_proxy):
    """An empty movement Points layer has no lanes or edited-frame bars."""
    viewer = make_napari_viewer_proxy()
    viewer.add_points(
        np.empty((0, 2)),
        properties={
            "edited": np.array([], dtype=bool),
            "individual": np.array([], dtype=str),
        },
        metadata={POINTS_LAYER_KEY: True},
    )
    edit_timeline_widget = EditTimelineWidget(viewer)

    edit_timeline_widget.set_show_individuals(True)

    assert list(edit_timeline_widget.ax.get_yticks()) == []
    assert edit_timeline_widget._bars == []


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
    edit_timeline_widget = EditTimelineWidget(
        loader_with_two_edited_individuals.viewer
    )
    single = to_rgba(edit_timeline_widget._edit_bar_color)

    assert _bar_colors(edit_timeline_widget) == [single, single]

    edit_timeline_widget.set_show_individuals(True)
    layer = edit_timeline_widget.active_layer
    # Paint each individual a known, distinctive colour; then redraw.
    # Bars must come back as these exact colours
    known = {"id_0": (1.0, 0.0, 0.0, 1.0), "id_1": (0.0, 0.0, 1.0, 1.0)}
    layer.face_color = np.array(
        [known[ind] for ind in layer.properties["individual"]]
    )
    edit_timeline_widget._redraw_bars()

    # id_0 edited on frame 2; id_1 on frames 2 and 5, so  red once, blue twice.
    assert sorted(_bar_colors(edit_timeline_widget)) == pytest.approx(
        sorted([known["id_0"], known["id_1"], known["id_1"]])
    )

    edit_timeline_widget.set_show_individuals(False)
    assert _bar_colors(edit_timeline_widget) == [single, single]


def test_bar_color_lookup_falls_back_without_individual_property(
    loader_with_edited_point,
):
    """Falls back to the shared edit colour if there's no individual data."""
    edit_timeline_widget = EditTimelineWidget(loader_with_edited_point.viewer)
    edit_timeline_widget._show_individuals = True
    edit_timeline_widget.active_layer = Mock(properties={})

    color_of = edit_timeline_widget._bar_color_lookup()

    assert color_of("id_0") == edit_timeline_widget._edit_bar_color
