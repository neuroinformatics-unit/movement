"""Test the napari plugin meta widget."""

from movement.napari.meta_widget import MovementMetaWidget


def test_meta_widget_instantiation(make_napari_viewer_proxy):
    """Test that the meta widget can be properly instantiated."""
    viewer = make_napari_viewer_proxy()
    meta_widget = MovementMetaWidget(viewer)

    # Should have 2 widgets now: DataLoader and LegendWidget
    assert len(meta_widget.collapsible_widgets) == 2

    first_widget = meta_widget.collapsible_widgets[0]
    assert first_widget._text == "Load tracked data"
    assert first_widget.isExpanded()

    # Check second widget is legend
    second_widget = meta_widget.collapsible_widgets[1]
    assert second_widget._text == "Color legend"

    # Check that loader and legend attributes are set
    # (they are collapsible wrappers)
    assert meta_widget.loader is not None
    assert meta_widget.legend is not None
