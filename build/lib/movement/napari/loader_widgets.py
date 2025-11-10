"""Widgets for loading movement datasets from file."""

from pathlib import Path

import numpy as np
import pandas as pd
from napari.components.dims import RangeTuple
from napari.layers import Image, Points, Shapes, Tracks
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

from movement.io import load_bboxes, load_poses
from movement.napari.convert import ds_to_napari_layers
from movement.napari.layer_styles import BoxesStyle, PointsStyle, TracksStyle
from movement.utils.logging import logger

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

        # Connect frame slider range update to layer events
        for action_str in ["inserted", "removed"]:
            getattr(self.viewer.layers.events, action_str).connect(
                self._update_frame_slider_range,
            )

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
        """Load the file and add as a Points and Tracks layers."""
        # Get input data
        self.fps = self.fps_spinbox.value()
        self.source_software = self.source_software_combo.currentText()
        self.file_path = self.file_path_edit.text()
        self.file_name = Path(self.file_path).name

        # Check if the file path is empty
        if not self.file_path:
            show_warning("No file path specified.")
            return

        # Format data for napari layers
        self._format_data_for_layers()
        logger.info("Converted dataset to a napari Tracks array.")
        logger.debug(f"Tracks array shape: {self.data.shape}")
        if self.data_bboxes is not None:
            logger.debug(f"Shapes array shape: {self.data_bboxes.shape}")

        # Set property to color points, tracks and boxes by
        self._set_common_color_property()

        # Set text property for points layer
        self._set_text_property()

        # Add the data as a points and a tracks layers,
        # and a boxes layer if the dataset is a bounding boxes one
        self._add_points_layer()
        self._add_tracks_layer()
        if self.data_bboxes is not None:
            self._add_boxes_layer()

        # Set the frame slider position
        self._set_initial_state()

    def _format_data_for_layers(self):
        """Extract data required for the creation of the napari layers."""
        # Load data as a movement dataset
        loader = (
            load_poses
            if self.source_software in SUPPORTED_POSES_FILES
            else load_bboxes
        )
        ds = loader.from_file(self.file_path, self.source_software, self.fps)

        # Convert to napari arrays
        self.data, self.data_bboxes, self.properties = ds_to_napari_layers(ds)

        # Find rows that do not contain NaN values
        self.data_not_nan = ~np.any(np.isnan(self.data), axis=1)

    def _set_common_color_property(self):
        """Set the color property for the Points and Tracks layers.

        The color property is set to "individual" by default,
        If the dataset contains only one individual and "keypoint"
        is defined as a property, the color property is set to "keypoint".
        """
        color_property = "individual"
        n_individuals = len(self.properties["individual"].unique())
        if n_individuals == 1 and "keypoint" in self.properties:
            color_property = "keypoint"
        self.color_property = color_property

        # Factorize the color property in the dataframe
        # (required for Tracks and Shapes layer)
        codes, _ = pd.factorize(self.properties[self.color_property])
        self.color_property_factorized = self.color_property + "_factorized"
        self.properties[self.color_property_factorized] = codes

    def _set_text_property(self):
        """Set the text property for the Points layer."""
        text_property = "individual"
        if (
            "keypoint" in self.properties
            and len(self.properties["keypoint"].unique()) > 1
        ):
            text_property = "keypoint"
        self.text_property = text_property

    def _set_initial_state(self):
        """Set slider at first frame and last points layer as active."""
        # Set slider to first frame so that first view is not cluttered
        # with tracks
        default_current_step = self.viewer.dims.current_step
        self.viewer.dims.current_step = (0,) + default_current_step[2:]

        # Set last loaded points layer as active
        self.viewer.layers.selection.active = [
            ly for ly in self.viewer.layers if isinstance(ly, Points)
        ][-1]

    def _add_points_layer(self):
        """Add the tracked data to the viewer as a Points layer."""
        # Define style for points layer
        points_style = PointsStyle(name=f"points: {self.file_name}")
        points_style.set_text_by(property=self.text_property)
        points_style.set_color_by(
            property=self.color_property,
            properties_df=self.properties,
        )

        # Add data as a points layer with metadata
        # (max_frame_idx is used to set the frame slider range)
        self.points_layer = self.viewer.add_points(
            self.data[self.data_not_nan, 1:],
            properties=self.properties.iloc[self.data_not_nan, :],
            metadata={"max_frame_idx": max(self.data[:, 1])},
            **points_style.as_kwargs(),
        )

        logger.info("Added tracked dataset as a napari Points layer.")

    def _add_tracks_layer(self):
        """Add the tracked data to the viewer as a Tracks layer."""
        # Define style for tracks layer
        tracks_style = TracksStyle(
            name=f"tracks: {self.file_name}",
            tail_length=int(max(self.data[:, 1])),
            # Set the tail length to the number of frames in the data.
            # If the value is over 300, it sets the maximum
            # tail_length in the slider to the value passed.
            # It also affects the head_length slider.
        )

        tracks_style.set_color_by(property=self.color_property_factorized)

        # Add data as a tracks layer
        self.viewer.add_tracks(
            self.data[self.data_not_nan, :],
            properties=self.properties.iloc[self.data_not_nan, :],
            metadata={"max_frame_idx": max(self.data[:, 1])},
            **tracks_style.as_kwargs(),
        )
        logger.info("Added tracked dataset as a napari Tracks layer.")

    def _add_boxes_layer(self):
        """Add bounding boxes data to the viewer as a Shapes layer."""
        # Define style for boxes layer
        bboxes_style = BoxesStyle(
            name=f"boxes: {self.file_name}",
        )
        bboxes_style.set_text_by(property=self.text_property)
        bboxes_style.set_color_by(
            property=self.color_property,
            properties_df=self.properties,
        )

        # Add bounding boxes data as a shapes layer
        self.bboxes_layer = self.viewer.add_shapes(
            self.data_bboxes[self.data_not_nan, :, 1:],
            properties=self.properties.iloc[self.data_not_nan, :],
            metadata={"max_frame_idx": max(self.data_bboxes[:, 0, 1])},
            **bboxes_style.as_kwargs(),
        )
        logger.info("Added tracked dataset as a napari Shapes layer.")

    def _update_frame_slider_range(self):
        """Check the frame slider range and update it if necessary.

        This is required because if the data loaded starts or ends
        with all NaN values, the frame slider range will not reflect
        the full range of frames.
        """
        # Only update the frame slider range if there are layers
        # that are Points, Tracks, Image or Shapes
        list_layers = [
            ly
            for ly in self.viewer.layers
            if isinstance(ly, Points | Tracks | Image | Shapes)
        ]
        if len(list_layers) > 0:
            # Get the maximum frame index from all candidate layers
            max_frame_idx = max(
                # For every layer, get max_frame_idx metadata if it exists,
                # else deduce it from the data shape
                [
                    getattr(ly, "metadata", {}).get(
                        "max_frame_idx", ly.data.shape[0] - 1
                    )
                    if not isinstance(ly, Shapes)
                    # Napari stores shapes layer data as a list of 2D arrays
                    # instead of a 3D array, so we can't use data.shape here
                    else getattr(ly, "metadata", {}).get(
                        "max_frame_idx", len(ly.data) - 1
                    )
                    for ly in list_layers
                ]
            )

            # If the frame slider range is not set to the full range of frames,
            # update it.
            if (self.viewer.dims.range[0].stop != max_frame_idx) or (
                int(self.viewer.dims.range[0].start) != 0
            ):
                self.viewer.dims.range = (
                    RangeTuple(start=0.0, stop=max_frame_idx, step=1.0),
                ) + self.viewer.dims.range[1:]

    @staticmethod
    def _enable_layer_tooltips():
        """Toggle on tooltip visibility for napari layers.

        This nicely displays the layer properties as a tooltip
        when hovering over the layer in the napari viewer.
        """
        settings = get_settings()
        settings.appearance.layer_tooltip_visibility = True
