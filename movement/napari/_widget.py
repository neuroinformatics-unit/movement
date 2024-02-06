from brainglobe_utils.qtpy.collapsible_widget import CollapsibleWidgetContainer
from napari.viewer import Viewer

from movement.napari.loader_widgets import FileLoader


class MovementWidgets(CollapsibleWidgetContainer):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__()

        self.add_widget(
            FileLoader(napari_viewer, parent=self),
            collapsible=True,
            widget_title="Load files",
        )

        self.file_loader = self.collapsible_widgets[0]
        # expand FileLoader widget by default
        self.file_loader.expand()
