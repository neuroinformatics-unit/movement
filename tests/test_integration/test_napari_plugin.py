import pytest
from qtpy.QtWidgets import QPushButton, QWidget

from movement.napari.meta_widget import MovementMetaWidget


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


def test_hello_button(loader_widget, capsys):
    """Test that the hello button works as expected."""
    hello_button = loader_widget.findChildren(QPushButton)[0]
    assert hello_button.text() == "Say hello"
    hello_button.click()
    captured = capsys.readouterr()
    assert "INFO: Hello, world!" in captured.out
