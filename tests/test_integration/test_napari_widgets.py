import pytest

from movement.napari.meta_widget import MovementMetaWidget


@pytest.fixture
def movement_meta_widget(make_napari_viewer) -> MovementMetaWidget:
    """Fixture to expose the MovementMetaWidget for testing.

    Simultaneously acts as a smoke test that the widget
    can be instantiated without crashing."""
    viewer = make_napari_viewer()
    return MovementMetaWidget(viewer)
