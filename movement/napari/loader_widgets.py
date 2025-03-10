"""Widgets for loading movement datasets from file."""

import logging
from pathlib import Path

import numpy as np
from napari.components.dims import RangeTuple
from napari.settings import get_settings
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

from movement.io import load_poses
from movement.napari.convert import poses_to_napari_tracks
from movement.napari.layer_styles import PointsStyle

logger = logging.getLogger(__name__)

# Allowed poses file suffixes for each supported source software
SUPPORTED_POSES_FILES = {
    "DeepLabCut": ["*.h5", "*.csv"],
    "LightningPose": ["*.csv"],
    "SLEAP": ["*.h5", "*.slp"],
}


class PosesLoader(QWidget):
    """Widget for loading movement poses datasets from file."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the loader widget."""
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
        self.source_software_combo.addItems(SUPPORTED_POSES_FILES.keys())
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
        file_suffixes = SUPPORTED_POSES_FILES[
            self.source_software_combo.currentText()
        ]

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open file containing predicted poses",
            filter=f"Poses files ({' '.join(file_suffixes)})",
        )

        # A blank string is returned if the user cancels the dialog
        if not file_path:
            return

        # Add the file path to the line edit (text field)
        self.file_path_edit.setText(file_path)

    def _on_load_clicked(self):
        """Load the file and add as a Points layer to the viewer."""
        fps = self.fps_spinbox.value()
        source_software = self.source_software_combo.currentText()
        file_path = self.file_path_edit.text()
        if file_path == "":
            show_warning("No file path specified.")
            return
        ds = load_poses.from_file(file_path, source_software, fps)

        self.data, self.props = poses_to_napari_tracks(ds)
        logger.info("Converted poses dataset to a napari Tracks array.")
        logger.debug(f"Tracks array shape: {self.data.shape}")

        self.file_name = Path(file_path).name
        self._add_points_layer()

    def _add_points_layer(self):
        """Add the predicted poses to the viewer as a Points layer."""
        # Find rows in data array that do not contain NaN values
        bool_not_nan = ~np.any(np.isnan(self.data), axis=1)

        # Style properties for the napari Points layer
        points_style = PointsStyle(
            name=f"poses: {self.file_name}",
            properties=self.props.iloc[bool_not_nan, :],
        )

        # Color the points by individual if there are multiple individuals
        # Otherwise, color by keypoint
        n_individuals = len(self.props["individual"].unique())
        points_style.set_color_by(
            prop="individual" if n_individuals > 1 else "keypoint"
        )
        # Add the points layer to the viewer
        self.viewer.add_points(
            self.data[bool_not_nan, 1:],  # columns:(track_id, frame_idx, y, x)
            **points_style.as_kwargs(),
        )
        # Ensure the frame slider reflects the total number of frames
        expected_frame_range = RangeTuple(
            start=0.0, stop=max(self.data[:, 1]), step=1.0
        )
        if self.viewer.dims.range[0] != expected_frame_range:
            self.viewer.dims.range = (
                expected_frame_range,
            ) + self.viewer.dims.range[1:]

        logger.info("Added poses dataset as a napari Points layer.")

    @staticmethod
    def _enable_layer_tooltips():
        """Toggle on tooltip visibility for napari layers.

        This nicely displays the layer properties as a tooltip
        when hovering over the layer in the napari viewer.
        """
        settings = get_settings()
        settings.appearance.layer_tooltip_visibility = True
