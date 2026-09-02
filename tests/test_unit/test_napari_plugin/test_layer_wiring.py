"""Test movement layer wiring outlives the meta-widget.

napari holds bound methods weakly, so callbacks owned by the
``DataLoader`` widget went dead once the widget was garbage collected
(e.g. after closing the movement panel), silently leaving the Points and
Tracks layers out of sync. These tests force that collection.
"""

import gc
import weakref

import numpy as np
import pytest
from napari.layers.base import ActionType

from movement.napari.layer_wiring import connect_viewer_callbacks


# TODO: add bboxes?
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


def test_point_edit_syncs_tracks_layer_after_widget_closed(
    orphan_viewer_and_layers,
):
    """Test that dragging a point updates the Tracks layer if widget gc-ed."""
    # Get layers of a loader widget that has been garbage-collected
    _, points_layer, tracks_layer = orphan_viewer_and_layers

    # Simulate point dragging
    edit_idx = 5
    edit_array = [100, 200]
    points_layer.data[edit_idx, 1:] = edit_array
    points_layer.events.data(
        value=points_layer.data,
        action=ActionType.CHANGED,
        data_indices=(edit_idx,),
        vertex_indices=((),),
    )

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


def test_connect_viewer_callbacks_is_idempotent(make_napari_viewer_proxy):
    """Test that wiring a viewer twice does not duplicate the callbacks."""
    # Wire the viewer callbacks once
    viewer = make_napari_viewer_proxy()
    connect_viewer_callbacks(viewer)

    # Count callbacks linked to the viewer.
    # (callbacks are linked to two events in `connect_viewer_callbacks`:
    # "inserted layers events" and "dimensions order events").
    n_inserted_callbacks = len(viewer.layers.events.inserted.callbacks)
    n_order_callbacks = len(viewer.dims.events.order.callbacks)

    # Connect the callbacks to the viewer again
    connect_viewer_callbacks(viewer)

    # The number of callbacks should not increase
    assert len(viewer.layers.events.inserted.callbacks) == n_inserted_callbacks
    assert len(viewer.dims.events.order.callbacks) == n_order_callbacks
