"""Test the callbacks in ``movement.napari.layer_wiring``.

Layer and viewer callbacks should remain active independently of the
``DataLoader`` widget lifetime, keeping the Points and Tracks layers in
sync after the movement panel is closed. These tests verify
that expectation, and that the viewer callbacks themselves behave.
"""

import gc
import weakref

import numpy as np
import pytest
from napari.components import ViewerModel
from napari.components.dims import RangeTuple
from napari.layers.base import ActionType

from movement.napari.layer_wiring import (
    MAX_FRAME_IDX_KEY,
    connect_viewer_callbacks,
    update_frame_slider_range,
)
from movement.napari.loader_widgets import DataLoader
from movement.napari.meta_widget import MovementMetaWidget


@pytest.fixture
def headless_napari_viewer():
    """Return a headless napari viewer model.

    Faster than make_napari_viewer_proxy because it does not
    require Qt viewer construction and teardown.
    """
    return ViewerModel()


@pytest.fixture
def orphan_viewer_and_layers(valid_poses_path_and_ds, loaded_data_loader):
    """Return viewer and layers of a loaded dataset whose widget is gc-ed."""
    # Get loader widget
    # (valid_poses_path_and_ds returns a 2-tuple
    # (out_path, valid_poses_dataset))
    loader = loaded_data_loader(*valid_poses_path_and_ds)

    # Get weak reference to loader widget
    loader_ref = weakref.ref(loader)

    # Get associated viewer and layers
    viewer, points_layer, tracks_layer = (
        loader.viewer,
        loader.points_layer,
        loader.tracks_layer,
    )

    # Delete widget and run garbage collection, as the
    # napari GUI eventually does after the movement panel is closed.
    del loader
    gc.collect()

    # Check there are no strong references to the loader anymore
    assert loader_ref() is None, "DataLoader was not garbage collected"

    return viewer, points_layer, tracks_layer


def simulate_point_drag(points_layer, edit_idx, new_position):
    """Move a point and emit the event napari emits after a drag."""
    points_layer.data[edit_idx, 1:] = new_position
    points_layer.events.data(
        value=points_layer.data,
        action=ActionType.CHANGED,
        data_indices=(edit_idx,),
        vertex_indices=((),),
    )


def test_point_edit_syncs_tracks_layer_after_widget_closed(
    orphan_viewer_and_layers,
):
    """Test that dragging a point updates the Tracks layer if widget gc-ed."""
    # Get layers of a loader widget that has been garbage-collected
    _, points_layer, tracks_layer = orphan_viewer_and_layers

    # Simulate point dragging
    edit_idx = 5
    edit_array = [100, 200]
    simulate_point_drag(points_layer, edit_idx, edit_array)

    # Check the edited boolean for the dragged point
    assert points_layer.properties["edited"][edit_idx]

    # Check the tracks layer holds the edited coordinates
    np.testing.assert_array_equal(tracks_layer.data[edit_idx, 2:], edit_array)


def test_point_removal_syncs_tracks_layer_after_widget_closed(
    orphan_viewer_and_layers,
):
    """Test that deleting a point removes Tracks layer row if widget gc-ed."""
    # Get layers of a loader widget that has been garbage-collected
    _, points_layer, tracks_layer = orphan_viewer_and_layers

    # Get tracks layer prior state
    removed_idx = 5
    n_rows = tracks_layer.data.shape[0]
    expected_next_row = tracks_layer.data[removed_idx + 1].copy()

    # Simulate point deletion
    points_layer.data = np.delete(points_layer.data, removed_idx, axis=0)
    points_layer.events.data(
        value=points_layer.data,
        action=ActionType.REMOVED,
        data_indices=(removed_idx,),
        vertex_indices=((),),
    )

    # Check tracks layer data has one less row
    assert tracks_layer.data.shape[0] == n_rows - 1

    # Check the data at the removed index has moved one row
    np.testing.assert_array_equal(
        tracks_layer.data[removed_idx], expected_next_row
    )


def test_rolling_axes_disables_editing_after_widget_closed(
    orphan_viewer_and_layers,
):
    """Test that rolling the axes still disables point editing."""
    # Check points layer is editable after widget is gc-ed
    viewer, points_layer, _ = orphan_viewer_and_layers
    assert points_layer.editable

    # Change order of dimensions in viewer
    # and corresponding change in points layer
    viewer.dims.order = (1, 0, 2)
    assert not points_layer.editable

    viewer.dims.order = (0, 1, 2)
    assert points_layer.editable


def test_3d_view_disables_editing_after_widget_closed(
    orphan_viewer_and_layers,
):
    """Test that switching to a 3D view still disables point editing.

    ``connect_viewer_callbacks`` wires ``update_points_layers_editable``
    to the ``ndisplay`` event as well as to ``order``.
    """
    # Check points layer is editable after widget is gc-ed
    viewer, points_layer, _ = orphan_viewer_and_layers
    assert points_layer.editable

    # Switch to a 3D view, where a drag could move a point to another frame
    viewer.dims.ndisplay = 3
    assert not points_layer.editable

    # Back to the default 2D view
    viewer.dims.ndisplay = 2
    assert points_layer.editable


def test_layer_wiring_survives_closing_metawidget(
    make_napari_viewer_proxy,
    valid_poses_path_and_ds,
    loaded_data_loader,
):
    """Test the layer wiring survives closing the movement panel.

    Unlike the tests above, which drop the ``DataLoader`` directly, this
    exercises the teardown path the napari GUI actually takes: clicking
    the panel's "x" calls ``QtViewerDockWidget.destroyOnClose``, which
    calls ``viewer.window.remove_dock_widget``.
    """
    # Instantiate and dock meta-widget
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    dock_widget = viewer.window.add_dock_widget(meta_widget, name="movement")

    # Get loader in meta_widget with data loaded
    loader = loaded_data_loader(
        *valid_poses_path_and_ds,
        loader=meta_widget.findChild(DataLoader),
    )

    # Get weak reference to the loader widget
    loader_ref = weakref.ref(loader)

    # Get layers
    points_layer, tracks_layer = loader.points_layer, loader.tracks_layer

    # Close the movement panel, as clicking its "x" does
    viewer.window.remove_dock_widget(dock_widget)
    del meta_widget, dock_widget, loader

    # Ensure the loader is gc-ed before asserting
    # (`remove_dock_widget` re-parents the meta-widget to None,
    # so the loader is reclaimed by ordinary Python gc once the
    # last reference to it is dropped).
    gc.collect()
    assert loader_ref() is None, "DataLoader was not garbage collected"

    # Check a point drag still syncs the Tracks layer
    edit_idx = 5
    edit_array = [100, 200]
    simulate_point_drag(points_layer, edit_idx, edit_array)

    assert points_layer.properties["edited"][edit_idx]
    np.testing.assert_array_equal(tracks_layer.data[edit_idx, 2:], edit_array)


def test_connect_viewer_callbacks_twice_does_not_duplicate(
    make_napari_viewer_proxy,
):
    """Test that wiring a viewer twice does not duplicate the callbacks."""
    # Wire the viewer callbacks once
    viewer = make_napari_viewer_proxy()
    connect_viewer_callbacks(viewer)

    # Count callbacks linked to the viewer, for each of the four events
    # `connect_viewer_callbacks` wires: layers "inserted" and "removed",
    # and dimensions "order" and "ndisplay".
    emitters = [
        viewer.layers.events.inserted,
        viewer.layers.events.removed,
        viewer.dims.events.order,
        viewer.dims.events.ndisplay,
    ]
    n_callbacks_per_emitter = [len(emitter.callbacks) for emitter in emitters]

    # Connect the callbacks to the viewer again
    connect_viewer_callbacks(viewer)

    # The number of callbacks should not increase
    assert [
        len(emitter.callbacks) for emitter in emitters
    ] == n_callbacks_per_emitter


# ---- update_frame_slider_range ------------------------------------------
# We ensure movement layers get their full frame span back, even if there are
# frames with all nan data. Layers movement did not create should keep the
# range napari sets.


# The movement layers below all span 100 frames (indices 0 to 99);
# what differs between tests is how many leading or trailing frames
# NaN-trimming dropped.
N_FRAMES = 100


def get_viewer_with_trimmed_points(
    viewer, first_frame_w_data, last_frame_w_data=None
):
    """Get a napari viewer with a movement-like NaN-trimmed Points layer.

    The layer holds points for ``first_frame_w_data..last_frame_w_data`` only,
    assuming the remaining leading or trailing frames are all-NaN and dropped.

    It declares the true last frame index (``N_FRAMES - 1``) in its metadata
    the way ``DataLoader`` does.
    """
    if last_frame_w_data is None:
        last_frame_w_data = N_FRAMES - 1

    frames = np.arange(first_frame_w_data, last_frame_w_data + 1)

    return viewer.add_points(
        # one point per frame, with columns (frame, y, x)
        np.column_stack((frames, np.full((frames.size, 2), 10.0))),
        metadata={MAX_FRAME_IDX_KEY: N_FRAMES - 1},
    )


@pytest.mark.parametrize(
    "first_frame_w_data, last_frame_w_data",
    [
        pytest.param(
            51,
            N_FRAMES - 1,
            id="leading_nans",
        ),
        pytest.param(
            0,
            49,
            id="trailing_nans",
        ),
    ],
)
def test_frame_slider_range_covers_nan_trimmed_frames(
    headless_napari_viewer,
    first_frame_w_data,
    last_frame_w_data,
):
    """Test frame slider update on a movement layer with all-NaN frames."""
    # The layer holds data for first_frame_w_data..last_frame_w_data only,
    # the remaining leading or trailing frames were dropped as all-NaN
    get_viewer_with_trimmed_points(
        headless_napari_viewer,
        first_frame_w_data,
        last_frame_w_data,
    )

    # check napari only sees the trimmed extent
    assert headless_napari_viewer.dims.range[0] == RangeTuple(
        first_frame_w_data, last_frame_w_data, 1.0
    )

    # call frame slider update
    update_frame_slider_range(headless_napari_viewer)

    # check the dropped frames are added back
    assert headless_napari_viewer.dims.range[0] == RangeTuple(
        0.0, N_FRAMES - 1, 1.0
    )


@pytest.mark.parametrize(
    "n_frames_video, expected_frame_range",
    [
        pytest.param(
            200,
            RangeTuple(0.0, 199.0, 1.0),  # matches video range
            id="video_longer_than_data",
        ),
        pytest.param(
            70,
            RangeTuple(0.0, N_FRAMES - 1, 1.0),  # matches movement data range
            id="video_shorter_than_data",
        ),
    ],
)
def test_frame_slider_range_w_non_movement_layers(
    headless_napari_viewer, n_frames_video, expected_frame_range
):
    """Test the frame slider with a non-movement layer of different lengths."""
    # Get a viewer with movement data spanning frame indices 20 to 49.
    # Full span is 0 to 99.
    get_viewer_with_trimmed_points(
        headless_napari_viewer,
        first_frame_w_data=20,
        last_frame_w_data=49,
    )

    # Add an Image layer with a mock video
    headless_napari_viewer.add_image(np.zeros((n_frames_video, 8, 8)))

    # Update the frame slider
    update_frame_slider_range(headless_napari_viewer)

    # The viewer range should match the largest span
    assert headless_napari_viewer.dims.range[0] == expected_frame_range


def test_frame_slider_range_ignores_row_count(headless_napari_viewer, rng):
    """Test that frame slider is not triggered by row count in layer data."""
    # A movement layer with data spanning all frames
    get_viewer_with_trimmed_points(
        headless_napari_viewer, first_frame_w_data=0
    )

    # Add a layer with 500 points, all of them in frame 0;
    # the 500 rows in this array should not trigger a frame range update
    headless_napari_viewer.add_points(
        np.column_stack((np.zeros(500), rng.random((500, 2))))
    )

    # Trigger frame slider update
    update_frame_slider_range(headless_napari_viewer)

    # The frame range should span the movement data only
    assert headless_napari_viewer.dims.range[0] == RangeTuple(
        0.0, N_FRAMES - 1, 1.0
    )
