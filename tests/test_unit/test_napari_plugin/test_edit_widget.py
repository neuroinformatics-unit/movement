"""Test the napari plugin edit widget."""

import pytest

from movement.napari.edit_widget import EditWidget


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
