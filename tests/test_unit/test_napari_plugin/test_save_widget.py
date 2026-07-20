"""Unit tests for the save widget in the napari plugin."""

import numpy as np
import xarray as xr
from qtpy.QtWidgets import QPushButton

from movement.napari.save_widget import DataSaver


def test_data_saver_widget_instantiation(make_napari_viewer_proxy):
    """Test that the save widget is properly instantiated."""
    data_saver_widget = DataSaver(make_napari_viewer_proxy())

    assert data_saver_widget.layout().rowCount() == 1
    assert isinstance(data_saver_widget.save_button, QPushButton)
    assert data_saver_widget.save_button.objectName() == "save_button"


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
            "movement_properties_with_nans": None,
            "movement_attrs": {},
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


def test_save_round_trip(
    tmp_path,
    valid_poses_path_and_ds,
    loaded_data_loader,
    mocker,
):
    """Test that saving a loaded dataset via the save widget reconstructs
    the original dataset in the netCDF file written to disk.
    """
    filepath, ds_loaded = valid_poses_path_and_ds
    loader = loaded_data_loader(filepath, ds_loaded)

    data_saver_widget = DataSaver(loader.viewer)
    loader.viewer.layers.selection.active = loader.points_layer

    out_path = tmp_path / "roundtrip.nc"
    mocker.patch(
        "movement.napari.save_widget.QFileDialog.getSaveFileName",
        return_value=(str(out_path), None),
    )

    data_saver_widget._on_save_clicked()

    assert out_path.exists()
    saved_ds = xr.open_dataset(out_path)
    xr.testing.assert_equal(saved_ds, ds_loaded)
