import pytest
from napari.settings import get_settings
from qtpy.QtWidgets import QComboBox, QLineEdit, QPushButton, QSpinBox, QWidget

from movement.napari._loader_widget import SUPPORTED_POSES_FILES, PosesLoader
from movement.napari._meta_widget import MovementMetaWidget


# -------------------- widget fixtures ---------------------------------------#
@pytest.fixture
def meta_widget(make_napari_viewer_proxy) -> MovementMetaWidget:
    """Fixture to expose the MovementMetaWidget for testing.
    Simultaneously acts as a smoke test that the widget
    can be instantiated without crashing.
    """
    viewer = make_napari_viewer_proxy()
    return MovementMetaWidget(viewer)


@pytest.fixture
def loader_widget(meta_widget) -> QWidget:
    """Fixture to expose the Loader widget for testing."""
    loader = meta_widget.loader.content()
    return loader


# -------------------- tests for widget instantiation ------------------------#
def test_meta_widget_instantiation(meta_widget):
    """Test that the meta widget is properly instantiated."""
    assert meta_widget is not None
    assert len(meta_widget.collapsible_widgets) == 1

    first_widget = meta_widget.collapsible_widgets[0]
    assert first_widget._text == "Load poses"
    assert first_widget.isExpanded()


def test_loader_widget_instantiation(loader_widget):
    """Test that the loader widget is properly instantiated."""
    assert loader_widget is not None
    assert loader_widget.layout().rowCount() == 4

    # Make sure the all rows except last start with lowercase text
    # which ends with a semicolon
    for i in range(loader_widget.layout().rowCount() - 1):
        label = loader_widget.layout().itemAt(i, 0).widget()
        assert label.text().islower()
        assert label.text().endswith(":")

    # Make sure that the source software combo box is populated
    source_software_combo = loader_widget.findChildren(QComboBox)[0]
    assert source_software_combo.count() == len(SUPPORTED_POSES_FILES)

    # Test that the default fps is 30
    fps_spinbox = loader_widget.findChildren(QSpinBox)[0]
    assert fps_spinbox.value() == 30

    # Make sure that the line edit for file path is empty
    file_path_edit = loader_widget.findChildren(QLineEdit)[-1]
    assert file_path_edit.text() == ""

    # Make sure that the first button is a "Browse" button
    browse_button = loader_widget.findChildren(QPushButton)[0]
    assert browse_button.text() == "Browse"

    # Make sure that the last row is a "Load" button
    load_button = loader_widget.findChildren(QPushButton)[-1]
    assert load_button.text() == "Load"

    # Make sure that layer tooltips are enabled
    assert get_settings().appearance.layer_tooltip_visibility is True


# -------------------- tests for callbacks -----------------------------------#
def test_browse_button_calls_on_browse_clicked(
    make_napari_viewer_proxy, mocker
):
    """Test that clicking the 'Browse' button calls the right function.

    Here we have to create a new Loader widget after mocking the method.
    We cannot reuse the existing widget fixture because then it would be too
    late to mock (the widget has already "decided" which method to call).
    """
    mock_method = mocker.patch(
        "movement.napari._loader_widget.PosesLoader._on_browse_clicked"
    )
    loader = PosesLoader(make_napari_viewer_proxy)
    browse_button = loader.findChildren(QPushButton)[0]
    browse_button.click()
    mock_method.assert_called_once()


def test_load_button_calls_on_load_clicked(make_napari_viewer_proxy, mocker):
    """Test that clicking the 'Load' button calls the right function.

    Here we also have to create a new Loader widget after mocking the method,
    as in the previous test.
    """
    mock_method = mocker.patch(
        "movement.napari._loader_widget.PosesLoader._on_load_clicked"
    )
    loader = PosesLoader(make_napari_viewer_proxy)
    load_button = loader.findChildren(QPushButton)[-1]
    load_button.click()
    mock_method.assert_called_once()


def test_on_load_clicked_without_file_path(loader_widget, capsys):
    """Test that clicking 'Load' without a file path shows a warning."""
    loader_widget._on_load_clicked()
    captured = capsys.readouterr()
    assert "No file path specified." in captured.out


def test_on_load_clicked_with_valid_file_path(loader_widget, caplog):
    """Test clicking 'Load' with a valid file path.

    This test checks that the `_on_load_clicked` method causes the following:
    - creates the `data`, `props`, and `file_name` attributes
    - emits a series of expected log messages
    - adds a Points layer to the viewer (with the correct name)
    - sets the playback fps to the correct value
    """
    # Set the file path to a valid file
    file_path = pytest.DATA_PATHS.get("DLC_single-wasp.predictions.h5")
    loader_widget.file_path_edit.setText(file_path.as_posix())

    # Set the fps to 60
    loader_widget.fps_spinbox.setValue(60)

    # Call the _on_load_clicked method (pretend the user clicked "Load")
    loader_widget._on_load_clicked()

    # Check that class attributes have been created
    assert loader_widget.file_name == file_path.name
    assert loader_widget.data is not None
    assert loader_widget.props is not None

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
    points_layer = loader_widget.viewer.layers[0]
    assert points_layer.name == f"poses: {file_path.name}"

    # Check that the playback fps was set correctly
    assert get_settings().application.playback_fps == 60
