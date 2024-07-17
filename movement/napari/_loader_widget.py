from napari.utils.notifications import show_info
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QFormLayout,
    QPushButton,
    QWidget,
)


class Loader(QWidget):
    """Widget for loading data from files."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the loader widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())
        # Create widgets
        self._create_hello_widget()

    def _create_hello_widget(self):
        """Create the hello widget.

        This widget contains a button that, when clicked, shows a greeting.
        """
        hello_button = QPushButton("Say hello")
        hello_button.clicked.connect(self._on_hello_clicked)
        self.layout().addRow("Greeting", hello_button)

    def _on_hello_clicked(self):
        """Show a greeting."""
        show_info("Hello, world!")
