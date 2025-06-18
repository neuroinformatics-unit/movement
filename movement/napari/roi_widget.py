"""Widget for defining regions of interest (ROIs)."""

from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QFormLayout,
    QWidget,
)


class ROIWidget(QWidget):
    """Widget for defining regions of interest (ROIs)."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the ROI widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())
