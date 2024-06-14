from brainglobe_utils.qtpy.collapsible_widget import CollapsibleWidgetContainer
from napari.viewer import Viewer

from movement.napari.loader_widget import Loader


class MovementMetaWidget(CollapsibleWidgetContainer):
    """The widget to rule all movement napari widgets.

    This is a container of collapsible widgets, each responsible
    for handing specific tasks in the movement napari workflow.
    """

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__()

        self.add_widget(
            Loader(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Load",
        )

        self.loader = self.collapsible_widgets[0]
        self.loader.expand()  # expand the loader widget by default
