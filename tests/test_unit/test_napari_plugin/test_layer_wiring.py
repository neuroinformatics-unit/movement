"""Test that movement layer wiring outlives the meta-widget.

Layer and viewer callbacks should remain active independently of the
``DataLoader`` widget lifetime, keeping the Points and Tracks layers in
sync after the movement panel is closed. These tests verify
that expectation.
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
def viewer_model():
    """Return a headless napari viewer model (no Qt required)."""
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
# napari derives ``dims.range`` from the world-coordinate union of all layer
# extents (``LayerList._ranges``), honouring each layer's scale and translate,
# and it does so before our callback runs. Our only job is to *widen* that
# range to cover frames hidden by the NaN-trimming of movement's own layers.
# These tests pin that contract: movement layers get their full frame span
# back, and layers movement did not create keep the range napari gave them.


def add_movement_points(viewer, n_frames=100, first_frame=51, **kwargs):
    """Add a Points layer mimicking a NaN-trimmed movement layer.

    The layer holds points for ``first_frame..n_frames - 1`` only, as if the
    leading frames were all-NaN and dropped, but declares the true last frame
    index in its metadata the way ``DataLoader`` does.
    """
    data = np.array(
        [[t, 10.0, 10.0] for t in range(first_frame, n_frames)],
    )
    return viewer.add_points(
        data, metadata={MAX_FRAME_IDX_KEY: n_frames - 1}, **kwargs
    )


def test_frame_slider_range_covers_nan_trimmed_frames(viewer_model):
    """A movement layer's dropped leading frames are added back."""
    add_movement_points(viewer_model)
    # napari only sees the trimmed extent
    assert viewer_model.dims.range[0] == RangeTuple(51.0, 99.0, 1.0)

    update_frame_slider_range(viewer_model)

    assert viewer_model.dims.range[0] == RangeTuple(0.0, 99.0, 1.0)


@pytest.mark.parametrize(
    "image_kwargs, expected",
    [
        pytest.param(
            {"scale": (2.0, 1.0, 1.0)},
            RangeTuple(0.0, 78.0, 2.0),
            id="scaled",
        ),
        pytest.param(
            {"translate": (1000.0, 0.0, 0.0)},
            RangeTuple(1000.0, 1039.0, 1.0),
            id="translated",
        ),
        pytest.param({}, RangeTuple(0.0, 39.0, 1.0), id="plain"),
    ],
)
def test_frame_slider_range_untouched_without_movement_layers(
    viewer_model, image_kwargs, expected
):
    """Layers movement did not create keep the range napari computed.

    A scaled or translated layer has a world-coordinate range that does not
    match its array indices. Overwriting it with raw indices would move the
    slider off the layer's actual frames.
    """
    viewer_model.add_image(np.zeros((40, 8, 8)), **image_kwargs)
    assert viewer_model.dims.range[0] == expected

    update_frame_slider_range(viewer_model)

    assert viewer_model.dims.range[0] == expected


def test_frame_slider_range_widens_without_disturbing_other_layers(
    viewer_model,
):
    """A movement layer is padded; a translated layer keeps its world range."""
    add_movement_points(viewer_model)
    viewer_model.add_image(np.zeros((40, 8, 8)), translate=(1000.0, 0.0, 0.0))

    update_frame_slider_range(viewer_model)

    # Start covers the movement layer's dropped frames, stop still reaches
    # the far end of the translated image.
    assert viewer_model.dims.range[0] == RangeTuple(0.0, 1039.0, 1.0)


def test_frame_slider_range_repadded_after_point_deletion(viewer_model):
    """Deleting points must not shrink the range below the true frame span.

    napari recomputes ``dims.range`` from the live extent on every data
    change, so removing the trailing points would otherwise cut the slider
    short.
    """
    points_layer = add_movement_points(viewer_model)
    update_frame_slider_range(viewer_model)

    points_layer.data = points_layer.data[:10]
    assert viewer_model.dims.range[0] == RangeTuple(51.0, 60.0, 1.0)

    update_frame_slider_range(viewer_model)

    assert viewer_model.dims.range[0] == RangeTuple(0.0, 99.0, 1.0)


def test_frame_slider_range_ignores_layers_without_frame_metadata(
    viewer_model,
):
    """Movement layers without a frame extent are not candidates.

    The ROI Shapes layers created by the regions widget carry no
    ``MAX_FRAME_IDX_KEY``: a region polygon has no frame span to contribute.
    """
    viewer_model.add_shapes(metadata={"movement_regions_layer": True})
    before = viewer_model.dims.range[0]

    update_frame_slider_range(viewer_model)

    assert viewer_model.dims.range[0] == before


def test_frame_slider_range_ignores_row_count_of_other_layers(
    viewer_model, rng
):
    """A layer's row count must not be read as a frame count.

    Only ``Image`` layers hold one frame per row of ``data``; in a Points,
    Tracks or Shapes layer a row is one point or shape, and any number of
    them can share a frame. Deriving a frame span from ``len(data)`` would
    stretch the slider far past the frames such a layer occupies.
    """
    add_movement_points(viewer_model, n_frames=100, first_frame=0)

    # 500 points, all of them within frame 0
    viewer_model.add_points(
        np.column_stack((np.zeros(500), rng.random((500, 2))))
    )

    update_frame_slider_range(viewer_model)

    assert viewer_model.dims.range[0] == RangeTuple(0.0, 99.0, 1.0)
