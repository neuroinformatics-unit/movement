"""Test the napari plugin meta widget."""

from movement.napari.meta_widget import MovementMetaWidget


def test_meta_widget_instantiation(make_napari_viewer_proxy):
    """Test that the meta widget can be properly instantiated."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)
    print(len(meta_widget.collapsible_widgets))
    assert len(meta_widget.collapsible_widgets) == 2

    first_widget = meta_widget.collapsible_widgets[0]

    second_widget = meta_widget.collapsible_widgets[1]

    assert first_widget._text == "Load tracked data"
    assert first_widget.isExpanded()
    # Check that the ROI widget is present and properly labeled
    assert second_widget._text == "Draw ROIs"
