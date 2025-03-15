"""The main napari widget for the ``movement`` package."""

from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer

from movement.napari.roi_widget import ROIDrawingWidget


class MovementMetaWidget(CollapsibleWidgetContainer):
    """The widget to rule all ``movement`` napari widgets.

    This is a container of collapsible widgets, each responsible
    for handing specific tasks in the movement napari workflow.
    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the meta-widget."""
        super().__init__()

        self.add_widget(
            DataLoader(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Load tracked data",
        )

        # Add ROI drawing widget
        self.add_widget(
            ROIDrawingWidget(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Draw ROIs",
        )

        # Store references to widgets
        self.loader = self.collapsible_widgets[0]
        self.roi_drawer = self.collapsible_widgets[1]

        # Expand the loader widget by default
        self.loader.expand()
