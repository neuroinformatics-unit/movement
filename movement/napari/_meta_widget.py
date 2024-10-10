"""The main napari widget for the ``movement`` package."""

from brainglobe_utils.qtpy.collapsible_widget import CollapsibleWidgetContainer
from napari.viewer import Viewer

from movement.napari._loader_widgets import PosesLoader


class MovementMetaWidget(CollapsibleWidgetContainer):
    """The widget to rule all ``movement`` napari widgets.

    This is a container of collapsible widgets, each responsible
    for handing specific tasks in the movement napari workflow.
    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the meta-widget."""
        super().__init__()

        self.add_widget(
            PosesLoader(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Load poses",
        )

        self.loader = self.collapsible_widgets[0]
        self.loader.expand()  # expand the loader widget by default
