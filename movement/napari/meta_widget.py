"""The main napari widget for the ``movement`` package."""

from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer

from movement.napari.legend_widget import LegendWidget
from movement.napari.loader_widgets import DataLoader


class MovementMetaWidget(CollapsibleWidgetContainer):
    """The widget to rule all ``movement`` napari widgets.

    This is a container of collapsible widgets, each responsible
    for handing specific tasks in the movement napari workflow.
    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the meta-widget."""
        super().__init__()

        # Add the data loader widget
        self.add_widget(
            DataLoader(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Load tracked data",
        )

        # Add the legend widget
        self.add_widget(
            LegendWidget(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Color legend",
        )

        self.loader = self.collapsible_widgets[0]
        self.legend = self.collapsible_widgets[1]
        self.loader.expand()  # expand the loader widget by default
