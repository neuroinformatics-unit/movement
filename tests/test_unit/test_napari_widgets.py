import pytest
from qtpy.QtWidgets import QWidget

from movement.napari.meta_widget import MovementMetaWidget


@pytest.fixture
def meta_widget(make_napari_viewer) -> MovementMetaWidget:
    """Fixture to expose the MovementMetaWidget for testing.

    Simultaneously acts as a smoke test that the widget
    can be instantiated without crashing."""
    viewer = make_napari_viewer()
    return MovementMetaWidget(viewer)


@pytest.fixture
def loader_widget(meta_widget) -> QWidget:
    """Fixture to expose the Loader widget for testing."""
    # content() gets the QWidget inside the CollapsibleWidget
    loader = meta_widget.loader.content()
    return loader


def test_meta_widget(meta_widget):
    """Test that the meta widget is properly instantiated."""
    assert meta_widget is not None
    assert len(meta_widget.collapsible_widgets) >= 1

    first_widget = meta_widget.collapsible_widgets[0]
    assert first_widget._text == "Load"
    assert first_widget.isExpanded()


def test_loader_widget(loader_widget):
    """Test that the loader widget is properly instantiated."""
    assert loader_widget is not None
    # Default values
    assert loader_widget.source_software_combo.currentText() == "SLEAP"
    assert loader_widget.fps_spinbox.value() == 50
    assert loader_widget.file_path_edit.text() == ""
    assert loader_widget.load_button.text() == "Load"
