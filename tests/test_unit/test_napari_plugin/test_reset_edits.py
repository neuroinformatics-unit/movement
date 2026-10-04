"""Reset dragged and deleted pose points to their loaded state."""

import numpy as np
import pytest
from napari.components import ViewerModel

from movement.napari import layer_wiring as wiring


@pytest.fixture
def reset_layers():
    """Create two tracks with two frames and independent loaded snapshots."""
    viewer = ViewerModel()
    data = np.array(
        [[0, 0, 1, 2], [0, 1, 3, 4], [1, 0, 5, 6], [1, 1, 7, 8]], dtype=float
    )
    props = {
        "confidence": np.array([0.1, 0.2, 0.3, 0.4]),
        "individual": np.array(["a", "a", "b", "b"]),
        "edited": np.array([True, False, False, False]),
    }
    points = viewer.add_points(
        data[:, 1:].copy(),
        properties=props,
        metadata={wiring.POINTS_LAYER_KEY: True},
    )
    tracks = viewer.add_tracks(data.copy(), properties=props)
    points.metadata[wiring.TRACKS_LAYER_KEY] = tracks
    wiring.set_point_symbol_by_edited(points)
    wiring.capture_points_baseline(points)
    points.events.data.connect(wiring.on_points_data_changed)
    return points, tracks, data, props, viewer


@pytest.mark.parametrize("frame", [None, 0, 1])
def test_reset_restores_deleted_points_and_keeps_other_edits(
    reset_layers, frame
):
    """Restore the selected frame without undoing another frame's edits."""
    points, tracks, original, props, viewer = reset_layers
    points.data[1, 1:] = [100, 200]
    points.events.data(action="changed", data_indices=(1,))
    points.remove([0, 2])
    wiring.reset_points_to_baseline(points, frame=frame)
    if frame in (None, 0):
        assert len(points.data) == 4
        np.testing.assert_array_equal(tracks.data[[0, 2]], original[[0, 2]])
    else:
        assert len(points.data) == 2
    if frame in (None, 1):
        np.testing.assert_array_equal(
            points.data[points.data[:, 0] == 1], original[[1, 3], 1:]
        )
    else:
        assert points.properties["edited"][1]
        np.testing.assert_array_equal(points.data[1, 1:], [100, 200])
    expected = points.data.copy()
    wiring.reset_points_to_baseline(points, frame=frame)
    np.testing.assert_array_equal(points.data, expected)


def test_reset_all_restores_loaded_properties_and_symbols(reset_layers):
    """Keep edits that were already in the loaded file, including styling."""
    points, tracks, original, props, viewer = reset_layers
    symbols = points.symbol.copy()
    points.remove([0, 1, 2])
    wiring.reset_points_to_baseline(points)
    np.testing.assert_array_equal(tracks.data, original)
    for key, values in props.items():
        np.testing.assert_array_equal(points.properties[key], values)
    np.testing.assert_array_equal(points.symbol, symbols)


def test_reset_controls_and_timeline(reset_layers, qtbot):
    """Buttons restore points and refresh timeline flags without OpenGL."""
    from movement.napari.edit_timeline_widget import (
        EditControlsWidget,
        EditTimelineWidget,
    )

    points, tracks, original, props, viewer = reset_layers
    controls = EditControlsWidget(napari_viewer=viewer)
    timeline = EditTimelineWidget(viewer)
    qtbot.addWidget(controls)
    qtbot.addWidget(timeline)
    assert controls.reset_frame_button.isEnabled()
    assert controls.reset_all_button.isEnabled()
    viewer.dims.order = (1, 0, 2)
    qtbot.waitUntil(lambda: not controls.reset_frame_button.isEnabled())
    assert controls.reset_all_button.isEnabled()
    viewer.dims.order = (0, 1, 2)
    qtbot.waitUntil(controls.reset_frame_button.isEnabled)
    points.data[1, 1:] = [100, 200]
    points.events.data(action="changed", data_indices=(1,))
    points.remove([2])
    viewer.dims.set_current_step(0, 1)
    controls.reset_frame_button.click()
    np.testing.assert_array_equal(tracks.data[1], original[1])
    assert len(points.data) == 3
    controls.reset_all_button.click()
    np.testing.assert_array_equal(tracks.data, original)
    np.testing.assert_array_equal(timeline._edited_frames, [0])
    assert timeline._removed_points == []
    viewer.layers.remove(tracks)
    qtbot.waitUntil(lambda: not controls.reset_all_button.isEnabled())


def test_reset_controls_without_loaded_pose(qtbot):
    """Reset is unavailable for unrelated points or an empty viewer."""
    from movement.napari.edit_timeline_widget import EditControlsWidget

    viewer = ViewerModel()
    controls = EditControlsWidget(napari_viewer=viewer)
    qtbot.addWidget(controls)
    assert not controls.reset_all_button.isEnabled()
    viewer.add_points([[0, 1, 2]])
    controls._update_reset_enabled()
    assert not controls.reset_frame_button.isEnabled()
    assert not controls.reset_all_button.isEnabled()


def test_loaded_baseline_roundtrip(valid_poses_dataset, tmp_path, qtbot):
    """Loading captures a baseline that restores missing data on export."""
    import xarray as xr

    from movement.napari.convert import napari_layers_to_ds
    from movement.napari.loader_widgets import DataLoader

    ds = valid_poses_dataset.copy(deep=True)
    ds.position.values.flat[0:2] = np.nan
    path = tmp_path / "loaded.nc"
    ds.to_netcdf(path)
    viewer = ViewerModel()
    loader = DataLoader(viewer)
    qtbot.addWidget(loader)
    loader.source_software_combo.setCurrentText("movement (netCDF)")
    loader.file_path_edit.setText(str(path))
    loader._on_load_clicked()
    points = loader.points_layer
    assert wiring.POINTS_BASELINE_KEY in points.metadata
    loaded = napari_layers_to_ds(
        points.data,
        points.properties,
        points.metadata[wiring.POINTS_PROPERTIES_KEY],
        ds.attrs,
    )
    points.data[0, 1:] += 50
    points.events.data(action="changed", data_indices=(0,))
    points.remove([1])
    wiring.reset_points_to_baseline(points)
    restored = napari_layers_to_ds(
        points.data,
        points.properties,
        points.metadata[wiring.POINTS_PROPERTIES_KEY],
        ds.attrs,
    )
    xr.testing.assert_allclose(restored.position, loaded.position)
    xr.testing.assert_allclose(restored.confidence, loaded.confidence)


def test_reset_in_docked_widget(
    make_napari_viewer_proxy,
    valid_poses_path_and_ds,
    loaded_data_loader,
    qtbot,
):
    """Exercise reset buttons through the real viewer and docked timeline."""
    from movement.napari.loader_widgets import DataLoader
    from movement.napari.meta_widget import MovementMetaWidget

    viewer = make_napari_viewer_proxy()
    widget = MovementMetaWidget(viewer)
    qtbot.addWidget(widget)
    loader = widget.findChild(DataLoader)
    path, ds = valid_poses_path_and_ds
    loaded_data_loader(path, ds, loader=loader)
    points = loader.points_layer
    tracks = points.metadata[wiring.TRACKS_LAYER_KEY]
    original = tracks.data.copy()
    points.data[0, 1:] += 50
    points.events.data(action="changed", data_indices=(0,))
    qtbot.waitUntil(widget.edit_controls.reset_frame_button.isEnabled)
    assert widget.edit_timeline_widget is not None
    viewer.dims.set_current_step(0, int(original[0, 1]))
    widget.edit_controls.reset_frame_button.click()
    np.testing.assert_array_equal(tracks.data, original)
    points.remove([0])
    widget.edit_controls.reset_all_button.click()
    np.testing.assert_array_equal(tracks.data, original)
    assert widget.edit_timeline_widget._removed_points == []
