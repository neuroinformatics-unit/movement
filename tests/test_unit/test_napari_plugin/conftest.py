"""Fixtures for testing the napari plugin."""

import pytest
from qtpy.QtWidgets import QWidget

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
def poses_loader_widget(meta_widget) -> QWidget:
    """Fixture to expose the PosesLoader widget for testing."""
    loader = meta_widget.loader.content()
    return loader
