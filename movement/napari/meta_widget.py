"""The main napari widget for the ``movement`` package."""

from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer

from movement.napari.edit_widget import EditWidget
from movement.napari.loader_widgets import DataLoader
from movement.napari.regions_widget import RegionsWidget
from movement.napari.save_widget import DataSaver


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

        # Add the Regions widget
        self.add_widget(
            RegionsWidget(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Define regions of interest",
        )

        # Add the Save widget
        self.add_widget(
            DataSaver(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Save tracked data",
        )

        loader_collapsible = self.collapsible_widgets[0]
        loader_collapsible.expand()  # expand the loader widget by default

        # The edit widget is a timeline, best shown full-width rather
        # than squeezed into this side panel, so it's docked separately
        # at the bottom instead of being added above as a collapsible
        # section.
        self.edit_widget = EditWidget(napari_viewer)
        napari_viewer.window.add_dock_widget(
            self.edit_widget, area="bottom", name="edited frames"
        )
