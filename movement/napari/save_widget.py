"""Widget for saving movement datasets from napari layers."""

from napari.layers import Points
from napari.utils.notifications import show_error, show_info
from napari.viewer import Viewer
from qtpy.QtWidgets import QFileDialog, QFormLayout, QPushButton, QWidget

from movement.napari.convert import napari_layers_to_ds
from movement.napari.loader_widgets import (
    ATTRS_KEY,
    PROPERTIES_KEY,
)
from movement.utils.logging import logger
from movement.validators.files import validate_file_path


class DataSaver(QWidget):
    """Widget for saving a tracked data layer to the native file format."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the data saver widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())
        self._create_save_button()

    def _create_save_button(self):
        """Create a button to save the selected points layer to file."""
        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("save_button")
        self.save_button.clicked.connect(self._on_save_clicked)
        self.layout().addRow(self.save_button)

    def _on_save_clicked(self):
        """Reconstruct a dataset from the selected layer and save it."""
        layer = self._get_points_layer_to_save()
        if layer is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save dataset as",
            filter="movement native format (*.nc)",
        )
        # A blank string is returned if the user cancels the dialog
        if not file_path:
            return
        if not file_path.lower().endswith(".nc"):
            file_path += ".nc"

        try:
            valid_path = validate_file_path(
                file_path, permission="w", suffixes={".nc"}
            )
            ds = napari_layers_to_ds(
                points_as_napari=layer.data,
                properties=layer.properties,
                properties_with_nans=layer.metadata[PROPERTIES_KEY],
                attrs=layer.metadata[ATTRS_KEY],
            )
            ds.to_netcdf(valid_path)
        except Exception as e:
            show_error(f"Failed to save dataset to '{file_path}': {e}")
            return

        logger.info(f"Saved dataset to '{valid_path}'.")
        show_info(f"Saved dataset to '{valid_path}'.")

    def _get_points_layer_to_save(self) -> Points | None:
        """Return the points layer to save, or None if selection is invalid."""
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, Points):
            show_error(
                "Please select a tracked data points layer in the "
                "layers list before saving."
            )
            return None
        if PROPERTIES_KEY not in layer.metadata:
            show_error(
                f"The layer '{layer.name}' was not loaded via the "
                "movement plugin and cannot be saved."
            )
            return None
        return layer
