"""Test the napari plugin meta widget."""

from movement.napari.meta_widget import MovementMetaWidget


def test_meta_widget_instantiation(make_napari_viewer_proxy):
    """Test that the meta widget can be properly instantiated."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)

    assert len(meta_widget.collapsible_widgets) == 2

    first_widget = meta_widget.collapsible_widgets[0]
    assert first_widget._text == "Load tracked data"
    assert first_widget.isExpanded()
    second = meta_widget.collapsible_widgets[1]
    assert second._text == "Export Shapes"
    assert not second.isExpanded()
