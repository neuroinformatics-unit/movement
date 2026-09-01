"""Tests that movement layer wiring outlives the widget that set it up.

napari holds bound methods weakly, so callbacks owned by the
``DataLoader`` widget went dead once the widget was garbage collected
(e.g. after closing the movement panel), silently leaving the Points and
Tracks layers out of sync. These tests force that collection.
"""

import gc

import numpy as np
import pytest
from napari.layers.base import ActionType


@pytest.fixture
def orphaned_layers(valid_poses_path_and_ds, loaded_data_loader):
    """Return the layers of a loaded dataset whose widget is gone.

    The ``DataLoader`` is dropped and the cyclic collector run, as the
    napari GUI eventually does after the movement panel is closed.
    """
    filepath, ds = valid_poses_path_and_ds
    loader = loaded_data_loader(filepath, ds)
    viewer, points_layer = loader.viewer, loader.points_layer
    tracks_layer = loader.tracks_layer

    del loader
    gc.collect()

    return viewer, points_layer, tracks_layer


def test_point_edit_syncs_tracks_layer_after_widget_is_gone(orphaned_layers):
    """Test that dragging a point still updates the Tracks layer."""
    _, points_layer, tracks_layer = orphaned_layers

    edit_idx = 5
    points_layer.data[edit_idx, 1:] = [100, 200]

    points_layer.events.data(
        value=points_layer.data,
        action=ActionType.CHANGED,
        data_indices=(edit_idx,),
        vertex_indices=((),),
    )

    assert points_layer.properties["edited"][edit_idx]
    np.testing.assert_array_equal(
        tracks_layer.data[edit_idx, 1:], points_layer.data[edit_idx]
    )


def test_point_removal_syncs_tracks_layer_after_widget_is_gone(
    orphaned_layers,
):
    """Test that deleting a point still removes its Tracks layer row."""
    _, points_layer, tracks_layer = orphaned_layers

    n_rows = tracks_layer.data.shape[0]
    removed_idx = 5
    expected_next_row = tracks_layer.data[removed_idx + 1].copy()
    points_layer.data = np.delete(points_layer.data, removed_idx, axis=0)

    points_layer.events.data(
        value=points_layer.data,
        action=ActionType.REMOVED,
        data_indices=(removed_idx,),
        vertex_indices=((),),
    )

    assert tracks_layer.data.shape[0] == n_rows - 1
    np.testing.assert_array_equal(
        tracks_layer.data[removed_idx], expected_next_row
    )


def test_editable_still_follows_axis_order_after_widget_is_gone(
    orphaned_layers,
):
    """Test that rolling the axes still disables point editing."""
    viewer, points_layer, _ = orphaned_layers
    assert points_layer.editable

    viewer.dims.order = (1, 0, 2)
    assert not points_layer.editable

    viewer.dims.order = (0, 1, 2)
    assert points_layer.editable
