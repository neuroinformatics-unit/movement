"""The main napari widget for the ``movement`` package."""

from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer

from movement.napari.loader_widgets import DataLoader
from movement.napari.roi_widget import RoiWidget


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

        # Add the ROI widget
        self.add_widget(
            RoiWidget(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Define ROIs",
        )

        loader_collapsible = self.collapsible_widgets[0]
        loader_collapsible.expand()  # expand the loader widget by default

        roi_collapsible = self.collapsible_widgets[1]
        roi_widget = roi_collapsible.content()
        # When ROI collapsible is expanded, initialise the ROI widget's
        # connection to a napari Shapes layer and ROI table model.
        roi_collapsible.toggled_signal_with_self.connect(
            lambda _, state: roi_widget.ensure_initialised() if state else None
        )
