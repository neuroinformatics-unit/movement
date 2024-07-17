import pytest
from qtpy.QtWidgets import QPushButton, QWidget

from movement.napari._loader_widget import Loader
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
    assert first_widget._text == "Load data"
    assert first_widget.isExpanded()


def test_loader_widget(loader_widget):
    """Test that the loader widget is properly instantiated."""
    assert loader_widget is not None
    assert loader_widget.layout().rowCount() == 1


def test_hello_button_calls_on_hello_clicked(make_napari_viewer_proxy, mocker):
    """Test that clicking the hello button calls _on_hello_clicked.

    Here we have to create a new Loader widget after mocking the method.
    We cannot reuse the existing widget fixture because then it would be too
    late to mock (the widget has already "decided" which method to call).
    """
    mock_method = mocker.patch(
        "movement.napari._loader_widget.Loader._on_hello_clicked"
    )
    loader = Loader(make_napari_viewer_proxy)
    hello_button = loader.findChildren(QPushButton)[0]
    hello_button.click()
    mock_method.assert_called_once()


def test_on_hello_clicked_outputs_message(loader_widget, capsys):
    """Test that _on_hello_clicked outputs the expected message."""
    loader_widget._on_hello_clicked()
    captured = capsys.readouterr()
    assert "INFO: Hello, world!" in captured.out
