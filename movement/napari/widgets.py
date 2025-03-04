"""Widgets for loading movement datasets from file."""

import logging
from pathlib import Path

import numpy as np
from napari.settings import get_settings
from napari.utils.colormaps import ensure_colormap
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from movement.io import load_bboxes, load_poses
from movement.napari.convert import movement_ds_to_napari_tracks

logger = logging.getLogger(__name__)

# Allowed file suffixes for each supported source software
SUPPORTED_POSES_FILES = {
    "DeepLabCut": ["h5", "csv"],
    "LightningPose": ["csv"],
    "SLEAP": ["h5", "slp"],
}

SUPPORTED_BBOXES_FILES = {
    "VIA-tracks": ["csv"],
}

SUPPORTED_DATA_FILES = {
    **SUPPORTED_POSES_FILES,
    **SUPPORTED_BBOXES_FILES,
}


class DataLoader(QWidget):
    """Widget for loading movement datasets from file."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the data loader widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())

        # Create widgets
        self._create_source_software_widget()
        self._create_fps_widget()
        self._create_file_path_widget()
        self._create_load_button()

        # Enable layer tooltips from napari settings
        self._enable_layer_tooltips()

    def _create_source_software_widget(self):
        """Create a combo box for selecting the source software."""
        self.source_software_combo = QComboBox()
        self.source_software_combo.setObjectName("source_software_combo")
        self.source_software_combo.addItems(SUPPORTED_DATA_FILES.keys())
        self.layout().addRow("source software:", self.source_software_combo)

    def _create_fps_widget(self):
        """Create a spinbox for selecting the frames per second (fps)."""
        self.fps_spinbox = QDoubleSpinBox()
        self.fps_spinbox.setObjectName("fps_spinbox")
        self.fps_spinbox.setMinimum(0.1)
        self.fps_spinbox.setMaximum(1000.0)
        self.fps_spinbox.setValue(1.0)
        self.fps_spinbox.setDecimals(2)
        # How much we increment/decrement when the user clicks the arrows
        self.fps_spinbox.setSingleStep(1)
        # Add a tooltip
        self.fps_spinbox.setToolTip(
            "Set the frames per second of the tracking data.\n"
            "This just affects the displayed time when hovering over a point\n"
            "(it doesn't set the playback speed)."
        )
        self.layout().addRow("fps:", self.fps_spinbox)

    def _create_file_path_widget(self):
        """Create a line edit and browse button for selecting the file path.

        This allows the user to either browse the file system,
        or type the path directly into the line edit.
        """
        # File path line edit and browse button
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setObjectName("file_path_edit")
        self.browse_button = QPushButton("Browse")
        self.browse_button.setObjectName("browse_button")
        self.browse_button.clicked.connect(self._on_browse_clicked)

        # Layout for line edit and button
        self.file_path_layout = QHBoxLayout()
        self.file_path_layout.addWidget(self.file_path_edit)
        self.file_path_layout.addWidget(self.browse_button)
        self.layout().addRow("file path:", self.file_path_layout)

    def _create_load_button(self):
        """Create a button to load the file and add layers to the viewer."""
        self.load_button = QPushButton("Load")
        self.load_button.setObjectName("load_button")
        self.load_button.clicked.connect(lambda: self._on_load_clicked())
        self.layout().addRow(self.load_button)

    def _on_browse_clicked(self):
        """Open a file dialog to select a file."""
        file_suffixes = (
            "*." + suffix
            for suffix in SUPPORTED_DATA_FILES[
                self.source_software_combo.currentText()
            ]
        )

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open file containing tracked data",
            filter=f"Valid data files ({' '.join(file_suffixes)})",
        )

        # A blank string is returned if the user cancels the dialog
        if not file_path:
            return

        # Add the file path to the line edit (text field)
        self.file_path_edit.setText(file_path)

    def _on_load_clicked(self):
        """Load the file and add as a Points layer to the viewer."""
        # Get data from user input
        fps = self.fps_spinbox.value()
        source_software = self.source_software_combo.currentText()
        file_path = self.file_path_edit.text()

        # Load data
        if file_path == "":
            show_warning("No file path specified.")
            return
        elif source_software in SUPPORTED_POSES_FILES:
            ds = load_poses.from_file(file_path, source_software, fps)
        elif source_software in SUPPORTED_BBOXES_FILES:
            ds = load_bboxes.from_file(file_path, source_software, fps)

        # Convert to napari Tracks array
        self.data, self.props = movement_ds_to_napari_tracks(ds)
        logger.info("Converted dataset to a napari Tracks array.")
        logger.debug(f"Tracks array shape: {self.data.shape}")

        # Add the data as a Points layer
        self.file_name = Path(file_path).name
        self._add_points_layer()

        # Add previous positions as a Tracks layer
        # self._add_tracks_layer()

    def _add_points_layer(self):
        """Add the tracked data to the viewer as a Points layer."""
        # Set property to color by
        color_prop = "individual"
        if (
            len(self.props["individual"].unique()) == 1
            and "keypoint" in self.props
        ):
            color_prop = "keypoint"

        def _sample_colormap(n: int, cmap_name: str) -> list[tuple]:
            cmap = ensure_colormap(cmap_name)
            samples = np.linspace(0, len(cmap.colors) - 1, n).astype(int)
            return [tuple(cmap.colors[i]) for i in samples]

        # Define color cycle
        n_colors = len(np.unique(self.props[color_prop]))
        color_cycle = _sample_colormap(n_colors, "turbo")

        # Add the first element of the data as a points layer
        slc_not_nan = ~np.any(np.isnan(self.data), axis=1)
        self.viewer.add_points(
            self.data[slc_not_nan, 1:],
            features=self.props.iloc[slc_not_nan, :],
            face_color=color_prop,
            face_color_cycle=color_cycle,
            border_width=0,
            text={
                "string": "{keypoint:}",
                "color": {"feature": color_prop, "colormap": color_cycle},
                "visible": False,
            },
            size=20,
            name=f"data: {self.file_name}",
        )
        logger.info("Added dataset as a napari Points layer.")

    def _add_tracks_layer(self):
        """Add the tracks data to the viewer as a Tracks layer."""
        # Style properties for the napari Tracks layer
        tracks_style = {
            "name": f"tracks: {self.file_name}",
            "properties": self.props,
        }

        # Add all data as a tracks layer
        self.viewer.add_tracks(self.data, **tracks_style)
        logger.info("Added dataset as a napari Tracks layer.")

    @staticmethod
    def _enable_layer_tooltips():
        """Toggle on tooltip visibility for napari layers.

        This nicely displays the layer properties as a tooltip
        when hovering over the layer in the napari viewer.
        """
        settings = get_settings()
        settings.appearance.layer_tooltip_visibility = True
