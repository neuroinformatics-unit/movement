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


@pytest.mark.parametrize(
    "make_layer",
    [
        pytest.param(
            lambda v: v.add_points(name="not from movement"),
            id="points_without_metadata",
        ),
        pytest.param(
            lambda v: v.add_points(
                name="not from movement",
                metadata={POINTS_PROPERTIES_KEY: None},
            ),
            id="points_with_properties_but_no_layer_key",
        ),
        pytest.param(
            lambda v: v.add_image(np.zeros((10, 10)), name="an image"),
            id="non_points_layer",
        ),
    ],
)
def test_save_button_disabled_for_invalid_layer(
    make_layer, make_napari_viewer_proxy
):
    """Test that selecting a layer that is not a movement points layer
    keeps the save button disabled with the default tooltip.

    The ``points_with_properties_but_no_layer_key`` case guards against
    treating the mere presence of ``POINTS_PROPERTIES_KEY`` (even if
    falsy/None) as a stand-in for "this layer was created by movement".
    """
    viewer = make_napari_viewer_proxy()
    data_saver_widget = DataSaver(viewer)

    viewer.layers.selection.active = make_layer(viewer)

    assert not data_saver_widget.save_button.isEnabled()
    assert data_saver_widget.save_button.toolTip() == DISABLED_TOOLTIP


def test_disabled_button_click_does_not_save(make_napari_viewer_proxy, mocker):
    """Test that clicking the disabled save button (no valid layer
    selected) never opens the file dialog.

    This guards the invariant that the button's enabled state is the
    sole gate on saving, so ``_on_save_clicked`` can safely assume the
    active layer is a valid movement points layer.
    """
    viewer = make_napari_viewer_proxy()
    data_saver_widget = DataSaver(viewer)

    viewer.layers.selection.active = viewer.add_image(
        np.zeros((10, 10)), name="an image"
    )

    mock_file_dialog = mocker.patch(
        "movement.napari.save_widget.QFileDialog.getSaveFileName"
    )

    # click() is a no-op on a disabled button, so the slot must not run
    data_saver_widget.save_button.click()

    mock_file_dialog.assert_not_called()


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
