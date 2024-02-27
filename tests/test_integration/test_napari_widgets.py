import pytest

from movement.napari._widget import MovementWidgets


@pytest.fixture
def movement_widget(make_napari_viewer) -> MovementWidgets:
    """Fixture to expose the MovementWidgets to the tests.

    Simultaneously acts as a smoke test that the widget
    can be instantiated without crashing."""
    viewer = make_napari_viewer()
    return MovementWidgets(viewer)
