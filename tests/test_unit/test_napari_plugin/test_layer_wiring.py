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
from movement.napari.loader_widgets import DataLoader
from movement.napari.meta_widget import MovementMetaWidget


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
