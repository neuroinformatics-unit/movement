import logging
from pathlib import Path

import numpy as np
from napari.utils.colormaps import ensure_colormap
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from movement.io import load_poses
from movement.napari.convert import ds_to_napari_tracks

logger = logging.getLogger(__name__)


def sample_colormap(n: int, cmap_name: str) -> list[tuple]:
    """Sample n equally-spaced colors from a napari colormap,
    including the endpoints."""
    cmap = ensure_colormap(cmap_name)
    samples = np.linspace(0, len(cmap.colors) - 1, n).astype(int)
    return [tuple(cmap.colors[i]) for i in samples]



class FileLoader(QWidget):
    """Widget for loading pose tracks from files into a napari viewer."""

    loader_func_map = {
        "DeepLabCut": load_poses.from_dlc_file,
        "LightingPose": load_poses.from_lp_file,
        "SLEAP": load_poses.from_sleap_file,
    }

    file_suffix_map = {
        "DeepLabCut": "Files containing predicted poses (*.h5 *.csv)",
        "LightingPose": "Files containing predicted poses (*.csv)",
        "SLEAP": "Files containing predicted poses (*.h5 *.slp)",
    }

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())
        # Create widgets
        self.create_source_software_widget()
        self.create_fps_widget()
        self.create_file_path_widget()

    def create_source_software_widget(self):
        """Create a combo box for selecting the source software."""
        self.source_software_combo = QComboBox()
        self.source_software_combo.addItems(
            ["SLEAP", "DeepLabCut", "LightningPose"]
        )
        self.layout().addRow("source software:", self.source_software_combo)

    def create_fps_widget(self):
        """Create a spinbox for selecting the frames per second (fps)."""
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setMinimum(1)
        self.fps_spinbox.setMaximum(1000)
        self.fps_spinbox.setValue(50)
        self.layout().addRow("fps:", self.fps_spinbox)

    def create_file_path_widget(self):
        """Create a line edit and browse button for selecting the file path.
        This allows the user to either browse the file system,
        or type the path directly into the line edit."""
        # File path line edit and browse button
        self.file_path_edit = QLineEdit()
        self.browse_button = QPushButton("browse")
        self.browse_button.clicked.connect(self.open_file_dialog)
        self.file_path_edit.returnPressed.connect(self.load_file_from_edit)
        # Layout for line edit and button
        self.file_path_layout = QHBoxLayout()
        self.file_path_layout.addWidget(self.file_path_edit)
        self.file_path_layout.addWidget(self.browse_button)
        self.layout().addRow("Pose file:", self.file_path_layout)

    def open_file_dialog(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.ExistingFile)
        # Allowed file suffixes based on the source software
        dlg.setNameFilter(
            self.file_suffix_map[self.source_software_combo.currentText()]
        )
        if dlg.exec_():
            file_paths = dlg.selectedFiles()
            # Set the file path in the line edit
            self.file_path_edit.setText(file_paths[0])
            # Load the file immediately after selection
            self.load_file(file_paths[0])

    def load_file_from_edit(self):
        # Load the file based on the path in the QLineEdit
        file_path = self.file_path_edit.text()
        self.load_file(file_path)

    def load_file(self, file_path):
        fps = self.fps_spinbox.value()
        source_software = self.source_software_combo.currentText()
        loader_func = self.loader_func_map[source_software]
        logger.debug(f"Using {loader_func}.")
        ds = loader_func(file_path, fps)

        self.data, self.props = ds_to_napari_tracks(ds)
        logger.info("Converted pose tracks to a napari Tracks array.")
        logger.debug(f"Tracks data shape: {self.data.shape}")

        self.file_name = Path(file_path).name
        self.add_layers()

    def add_layers(self):
        """Add the predicted pose tracks and keypoints to the napari viewer."""

        common_kwargs = {"visible": True, "blending": "translucent"}
        n_individuals = len(self.props["individual"].unique())
        n_keypoints = len(self.props["keypoint"].unique())
        color_by = "individual" if n_individuals > 1 else "keypoint"
        n_colors = n_individuals if color_by == "individual" else n_keypoints

        # kwargs for the napari Points layer
        points_kwargs = {
            **common_kwargs,
            "name": f"Keypoints - {self.file_name}",
            "properties": self.props,
            "symbol": "disc",
            "size": 10,
            "edge_width": 0,
            "face_color": color_by,
            "face_color_cycle": sample_colormap(n_colors, "turbo"),
            "face_colormap": "viridis",
            "text": {"string": color_by, "visible": False},
        }

        # Modify properties for the napari Tracks layer
        tracks_props = self.props.copy()
        # Track properties must be numeric, so convert str to categorical codes
        for col in ["individual", "keypoint"]:
            tracks_props[col] = tracks_props[col].astype("category").cat.codes
            logger.debug(f"{col} unique values {tracks_props[col].unique()}")

        # kwargs for the napari Tracks layer
        tracks_kwargs = {
            **common_kwargs,
            "name": f"Tracks - {self.file_name}",
            "properties": tracks_props,
            "tail_width": 5,
            "tail_length": 60,
            "head_length": 0,
            "color_by": color_by,
            "colormap": "turbo",
        }

        # Add the napari Tracks layer to the viewer
        self.viewer.add_tracks(self.data, **tracks_kwargs)

        # Add the napari Points layer to the viewer
        self.viewer.add_points(self.data[:, 1:], **points_kwargs)
