"""Unit tests for the save widget in the napari plugin."""

import numpy as np
import pytest
import xarray as xr
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QPushButton

from movement.napari.loader_widgets import (
    DATASET_ATTRS_KEY,
    POINTS_LAYER_KEY,
    POINTS_PROPERTIES_KEY,
)
from movement.napari.save_widget import (
    DISABLED_TOOLTIP,
    ENABLED_TOOLTIP,
    DataSaver,
)


def test_data_saver_widget_instantiation(make_napari_viewer_proxy):
    """Test that the save widget is properly instantiated."""
    data_saver_widget = DataSaver(make_napari_viewer_proxy())

    assert data_saver_widget.layout().rowCount() == 1
    assert isinstance(data_saver_widget.save_button, QPushButton)
    assert data_saver_widget.save_button.objectName() == "save_button"
    assert not data_saver_widget.save_button.isEnabled()
    assert data_saver_widget.save_button.toolTip() == DISABLED_TOOLTIP


def test_save_button_enabled_for_valid_points_layer(make_napari_viewer_proxy):
    """Test that selecting a valid movement points layer enables the save
    button and updates its tooltip.
    """
    viewer = make_napari_viewer_proxy()
    data_saver_widget = DataSaver(viewer)

    layer = viewer.add_points(
        name="points",
        metadata={
            POINTS_LAYER_KEY: True,
            POINTS_PROPERTIES_KEY: None,
            DATASET_ATTRS_KEY: {},
        },
    )
    viewer.layers.selection.active = layer

    assert data_saver_widget.save_button.isEnabled()
    assert data_saver_widget.save_button.toolTip() == ENABLED_TOOLTIP


def test_save_button_disabled_for_invalid_layer(make_napari_viewer_proxy):
    """Test that selecting a layer without movement metadata keeps the
    save button disabled with the default tooltip.
    """
    viewer = make_napari_viewer_proxy()
    data_saver_widget = DataSaver(viewer)

    layer = viewer.add_points(name="not from movement")
    viewer.layers.selection.active = layer

    assert not data_saver_widget.save_button.isEnabled()
    assert data_saver_widget.save_button.toolTip() == DISABLED_TOOLTIP


def test_save_clicked_without_points_layer_selected(
    make_napari_viewer_proxy, mocker
):
    """Test that clicking 'Save' without a valid layer selected shows
    an error and never opens the file dialog.
    """
    viewer = make_napari_viewer_proxy()
    data_saver_widget = DataSaver(viewer)

    mock_show_error = mocker.patch("movement.napari.save_widget.show_error")
    mock_file_dialog = mocker.patch(
        "movement.napari.save_widget.QFileDialog.getSaveFileName"
    )

    data_saver_widget._on_save_clicked()

    mock_show_error.assert_called_once()
    mock_file_dialog.assert_not_called()


def test_save_clicked_with_non_movement_points_layer(
    make_napari_viewer_proxy, mocker
):
    """Test that a Points layer without movement metadata is rejected."""
    viewer = make_napari_viewer_proxy()
    data_saver_widget = DataSaver(viewer)

    layer = viewer.add_points(name="not from movement")
    viewer.layers.selection.active = layer

    mock_show_error = mocker.patch("movement.napari.save_widget.show_error")
    mock_file_dialog = mocker.patch(
        "movement.napari.save_widget.QFileDialog.getSaveFileName"
    )

    data_saver_widget._on_save_clicked()

    mock_show_error.assert_called_once()
    mock_file_dialog.assert_not_called()


def test_save_clicked_with_properties_but_no_layer_key(
    make_napari_viewer_proxy, mocker
):
    """Test that a Points layer with a properties key but without the
    explicit POINTS_LAYER_KEY marker is still rejected.

    Guards against relying on the mere presence of
    ``POINTS_PROPERTIES_KEY`` (even if its value is falsy/None) as a
    stand-in for "this layer was created by movement".
    """
    viewer = make_napari_viewer_proxy()
    data_saver_widget = DataSaver(viewer)

    layer = viewer.add_points(
        name="not from movement",
        metadata={POINTS_PROPERTIES_KEY: None},
    )
    viewer.layers.selection.active = layer

    assert not data_saver_widget.save_button.isEnabled()

    mock_show_error = mocker.patch("movement.napari.save_widget.show_error")
    mock_file_dialog = mocker.patch(
        "movement.napari.save_widget.QFileDialog.getSaveFileName"
    )

    data_saver_widget._on_save_clicked()

    mock_show_error.assert_called_once()
    mock_file_dialog.assert_not_called()


def test_save_clicked_with_non_points_layer_selected(
    make_napari_viewer_proxy, mocker
):
    """Test that a non-Points layer selection is rejected."""
    viewer = make_napari_viewer_proxy()
    data_saver_widget = DataSaver(viewer)

    layer = viewer.add_image(np.zeros((10, 10)), name="an image")
    viewer.layers.selection.active = layer

    mock_show_error = mocker.patch("movement.napari.save_widget.show_error")

    data_saver_widget._on_save_clicked()

    mock_show_error.assert_called_once()


def test_save_clicked_cancelled_dialog(make_napari_viewer_proxy, mocker):
    """Test that cancelling the file dialog does not attempt to save."""
    viewer = make_napari_viewer_proxy()
    data_saver_widget = DataSaver(viewer)

    layer = viewer.add_points(
        name="points",
        metadata={
            POINTS_LAYER_KEY: True,
            POINTS_PROPERTIES_KEY: None,
            DATASET_ATTRS_KEY: {},
        },
    )
    viewer.layers.selection.active = layer

    mocker.patch(
        "movement.napari.save_widget.QFileDialog.getSaveFileName",
        return_value=("", None),
    )
    mock_to_netcdf = mocker.patch.object(xr.Dataset, "to_netcdf")

    data_saver_widget._on_save_clicked()

    mock_to_netcdf.assert_not_called()


@pytest.mark.parametrize(
    "dialog_name",
    [
        pytest.param("roundtrip.nc", id="with_suffix"),
        pytest.param("roundtrip", id="without_suffix"),
    ],
)
def test_save_round_trip(
    dialog_name,
    tmp_path,
    valid_poses_path_and_ds,
    loaded_data_loader,
    mocker,
):
    """Test that saving a loaded dataset via the save widget reconstructs
    the original dataset in the netCDF file written to disk.

    The ``dialog_name`` without a ".nc" suffix also covers the branch
    that appends it automatically.
    """
    filepath, ds_loaded = valid_poses_path_and_ds
    loader = loaded_data_loader(filepath, ds_loaded)

    data_saver_widget = DataSaver(loader.viewer)
    loader.viewer.layers.selection.active = loader.points_layer

    out_path = tmp_path / "roundtrip.nc"
    mocker.patch(
        "movement.napari.save_widget.QFileDialog.getSaveFileName",
        return_value=(str(tmp_path / dialog_name), None),
    )

    data_saver_widget._on_save_clicked()

    assert out_path.exists()
    saved_ds = xr.open_dataset(out_path)
    xr.testing.assert_equal(saved_ds, ds_loaded)
    assert saved_ds.attrs == loader.ds_attrs


def test_save_failure_shows_error(
    tmp_path,
    valid_poses_path_and_ds,
    loaded_data_loader,
    mocker,
):
    """Test that a failure during saving shows an error notification."""
    filepath, ds_loaded = valid_poses_path_and_ds
    loader = loaded_data_loader(filepath, ds_loaded)

    data_saver_widget = DataSaver(loader.viewer)
    loader.viewer.layers.selection.active = loader.points_layer

    out_path = tmp_path / "roundtrip.nc"
    mocker.patch(
        "movement.napari.save_widget.QFileDialog.getSaveFileName",
        return_value=(str(out_path), None),
    )
    mocker.patch.object(
        xr.Dataset, "to_netcdf", side_effect=OSError("disk full")
    )
    mock_show_error = mocker.patch("movement.napari.save_widget.show_error")

    data_saver_widget._on_save_clicked()

    mock_show_error.assert_called_once()
    assert not out_path.exists()


def test_close_event_disconnects_selection_signal(make_napari_viewer_proxy):
    """Test that closing the widget disconnects the layer selection
    callback, so it no longer reacts to further selection changes.
    """
    viewer = make_napari_viewer_proxy()
    data_saver_widget = DataSaver(viewer)

    data_saver_widget.closeEvent(QCloseEvent())

    layer = viewer.add_points(
        name="points",
        metadata={
            POINTS_LAYER_KEY: True,
            POINTS_PROPERTIES_KEY: None,
            DATASET_ATTRS_KEY: {},
        },
    )
    viewer.layers.selection.active = layer

    assert not data_saver_widget.save_button.isEnabled()
    assert data_saver_widget.save_button.toolTip() == DISABLED_TOOLTIP
