"""Unit tests for loader widgets in the napari plugin.

We instantiate the PosesLoader widget in each test instead of using a fixture.
This is because mocking widget methods would not work after the widget is
instantiated (the methods would have already been connected to signals).
"""

import pytest
from napari.settings import get_settings
from pytest import DATA_PATHS
from qtpy.QtWidgets import QComboBox, QLineEdit, QPushButton, QSpinBox

from movement.napari._loader_widgets import SUPPORTED_POSES_FILES, PosesLoader


# ------------------- tests for widget instantiation--------------------------#
def test_poses_loader_widget_instantiation(make_napari_viewer_proxy):
    """Test that the loader widget is properly instantiated."""
    # Instantiate the poses loader widget
    poses_loader_widget = PosesLoader(make_napari_viewer_proxy)

    # Check that the widget has the expected number of rows
    assert poses_loader_widget.layout().rowCount() == 4

    # Make sure the all rows except last start with lowercase text
    # which ends with a semicolon
    for i in range(poses_loader_widget.layout().rowCount() - 1):
        label = poses_loader_widget.layout().itemAt(i, 0).widget()
        assert label.text().islower()
        assert label.text().endswith(":")

    # Make sure that the source software combo box is populated
    source_software_combo = poses_loader_widget.findChildren(QComboBox)[0]
    assert source_software_combo.count() == len(SUPPORTED_POSES_FILES)

    # Test that the default fps is 30
    fps_spinbox = poses_loader_widget.findChildren(QSpinBox)[0]
    assert fps_spinbox.value() == 30

    # Make sure that the line edit for file path is empty
    file_path_edit = poses_loader_widget.findChildren(QLineEdit)[-1]
    assert file_path_edit.text() == ""

    # Make sure that the first button is a "Browse" button
    browse_button = poses_loader_widget.findChildren(QPushButton)[0]
    assert browse_button.text() == "Browse"

    # Make sure that the last row is a "Load" button
    load_button = poses_loader_widget.findChildren(QPushButton)[-1]
    assert load_button.text() == "Load"

    # Make sure that layer tooltips are enabled
    assert get_settings().appearance.layer_tooltip_visibility is True


# --------test connection between widget buttons and methods------------------#
def test_browse_button_calls_on_browse_clicked(
    make_napari_viewer_proxy, mocker
):
    """Test that clicking the 'Browse' button calls the right function."""
    mock_method = mocker.patch(
        "movement.napari._loader_widgets.PosesLoader._on_browse_clicked"
    )
    poses_loader_widget = PosesLoader(make_napari_viewer_proxy)
    browse_button = poses_loader_widget.findChildren(QPushButton)[0]
    browse_button.click()
    mock_method.assert_called_once()


def test_load_button_calls_on_load_clicked(make_napari_viewer_proxy, mocker):
    """Test that clicking the 'Load' button calls the right function."""
    mock_method = mocker.patch(
        "movement.napari._loader_widgets.PosesLoader._on_load_clicked"
    )
    poses_loader_widget = PosesLoader(make_napari_viewer_proxy)
    load_button = poses_loader_widget.findChildren(QPushButton)[-1]
    load_button.click()
    mock_method.assert_called_once()


# ------------------- tests for widget methods--------------------------------#
# In these tests we check if calling a widget method has the expected effects


@pytest.mark.parametrize(
    "file_path",
    [
        # valid file path
        str(DATA_PATHS.get("DLC_single-wasp.predictions.h5").parent),
        # empty string, simulate user canceling the dialog
        "",
    ],
)
def test_on_browse_clicked(file_path, make_napari_viewer_proxy, mocker):
    """Test that the _on_browse_clicked method correctly sets the
    file path in the QLineEdit widget (file_path_edit).
    The file path is provided by mocking the return of the
    QFileDialog.getOpenFileName method.
    """
    # Instantiate the napari viewer and the poses loader widget
    viewer = make_napari_viewer_proxy()
    poses_loader_widget = PosesLoader(viewer)

    # Mock the QFileDialog.getOpenFileName method to return the file path
    mocker.patch(
        "movement.napari._loader_widgets.QFileDialog.getOpenFileName",
        return_value=(file_path, None),  # tuple(file_path, filter)
    )
    # Simulate the user clicking the 'Browse' button
    poses_loader_widget._on_browse_clicked()
    # Check that the file path edit text has been updated
    assert poses_loader_widget.file_path_edit.text() == file_path


def test_on_load_clicked_without_file_path(make_napari_viewer_proxy, capsys):
    """Test that clicking 'Load' without a file path shows a warning."""
    # Instantiate the napari viewer and the poses loader widget
    viewer = make_napari_viewer_proxy()
    poses_loader_widget = PosesLoader(viewer)
    # Call the _on_load_clicked method (pretend the user clicked "Load")
    poses_loader_widget._on_load_clicked()
    captured = capsys.readouterr()
    assert "No file path specified." in captured.out


def test_on_load_clicked_with_valid_file_path(
    make_napari_viewer_proxy, caplog
):
    """Test clicking 'Load' with a valid file path.

    This test checks that the `_on_load_clicked` method causes the following:
    - creates the `data`, `props`, and `file_name` attributes
    - emits a series of expected log messages
    - adds a Points layer to the viewer (with the expected name)
    - sets the playback fps to the specified value
    """
    # Instantiate the napari viewer and the poses loader widget
    viewer = make_napari_viewer_proxy()
    poses_loader_widget = PosesLoader(viewer)
    # Set the file path to a valid file
    file_path = pytest.DATA_PATHS.get("DLC_single-wasp.predictions.h5")
    poses_loader_widget.file_path_edit.setText(file_path.as_posix())

    # Set the fps to 60
    poses_loader_widget.fps_spinbox.setValue(60)

    # Call the _on_load_clicked method (pretend the user clicked "Load")
    poses_loader_widget._on_load_clicked()

    # Check that class attributes have been created
    assert poses_loader_widget.file_name == file_path.name
    assert poses_loader_widget.data is not None
    assert poses_loader_widget.props is not None

    # Check that the expected log messages were emitted
    expected_log_messages = [
        "Converted poses dataset to a napari Tracks array.",
        "Tracks array shape: ",
        "Added poses dataset as a napari Points layer.",
        "Set napari playback speed to ",
    ]
    for msg in expected_log_messages:
        assert any(msg in record.getMessage() for record in caplog.records)

    # Check that a Points layer was added to the viewer
    points_layer = poses_loader_widget.viewer.layers[0]
    assert points_layer.name == f"poses: {file_path.name}"

    # Check that the playback fps was set correctly
    assert get_settings().application.playback_fps == 60
