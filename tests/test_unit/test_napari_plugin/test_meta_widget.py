"""Test the napari plugin meta widget."""


# We use the meta_widget fixture from test_napa_plugin/conftest.py
def test_meta_widget_instantiation(meta_widget):
    """Test that the meta widget is properly instantiated."""
    assert meta_widget is not None
    assert len(meta_widget.collapsible_widgets) == 1

    first_widget = meta_widget.collapsible_widgets[0]
    assert first_widget._text == "Load poses"
    assert first_widget.isExpanded()
