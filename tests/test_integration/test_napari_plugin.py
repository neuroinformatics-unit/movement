import pytest
from qtpy.QtWidgets import QPushButton, QWidget

from movement.napari._loader_widget import PosesLoader
from movement.napari._meta_widget import MovementMetaWidget


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


def test_meta_widget(meta_widget):
    """Test that the meta widget is properly instantiated."""
    assert meta_widget is not None
    assert len(meta_widget.collapsible_widgets) == 1

    first_widget = meta_widget.collapsible_widgets[0]
    assert first_widget._text == "Load poses"
    assert first_widget.isExpanded()


def test_loader_widget(loader_widget):
    """Test that the loader widget is properly instantiated."""
    assert loader_widget is not None
    assert loader_widget.layout().rowCount() == 4


def test_load_button_calls_on_load_clicked(make_napari_viewer_proxy, mocker):
    """Test that clicking the 'Load' call the right function.

    Here we have to create a new Loader widget after mocking the method.
    We cannot reuse the existing widget fixture because then it would be too
    late to mock (the widget has already "decided" which method to call).
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
