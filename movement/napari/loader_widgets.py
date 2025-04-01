"""Widgets for loading movement datasets from file."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from napari import layers
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

from movement.io import load_bboxes, load_poses
from movement.napari.convert import ds_to_napari_tracks
from movement.napari.layer_styles import PointsStyle, TracksStyle

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

        # Connect widget methods to events
        self._connect_methods_to_events()

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

    def _connect_methods_to_events(self):
        """Connect widget methods to napari events."""
        # Connect method to layer deletion event
        if hasattr(self.viewer, "layers"):
            self.viewer.layers.events.removed.connect(
                self._on_layer_deleted, ref=True
            )

        # Connect method to frame slider change event
        self.viewer.dims.events.current_step.connect(
            self._on_frame_slider_changed, ref=True
        )

    def _on_layer_deleted(self):
        """Check the frame slider range when a layer is deleted."""
        self._check_frame_slider_range()

    def _on_frame_slider_changed(self, event):
        """Show previous N points when the frame slider position changes."""
        self._show_N_prior_markers(event)
        # pass

    def _check_frame_slider_range(self):
        """Check the frame slider range and update it if necessary.

        This is required because if the data loaded starts or ends
        with all NaN values, the frame slider range will not reflect
        the full range of frames.
        """
        # If no layers are loaded or no Points layers are loaded, do nothing
        if not self.viewer.layers or not any(
            isinstance(ly, layers.Points) for ly in self.viewer.layers
        ):
            return

        # Get the maximum frame index from all loaded layers
        max_frame_idx = max(
            [
                ly.metadata["max_frame_idx"]
                for ly in self.viewer.layers
                if isinstance(ly, layers.Points)
                and hasattr(ly, "metadata")
                and "max_frame_idx" in ly.metadata
            ]
        )

        # If the frame slider range is not set to the full range of frames,
        # update it.
        # Note: the start frame may be different from 0 if all the data
        # at the first frame is NaN
        if (self.viewer.dims.range[0].stop != max_frame_idx) or (
            int(self.viewer.dims.range[0].start) != 0
        ):
            self.viewer.dims.range = (
                RangeTuple(start=0.0, stop=max_frame_idx, step=1.0),
            ) + self.viewer.dims.range[1:]

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
        self.file_name = Path(file_path).name

        # Check if the file path is empty
        if not file_path:
            show_warning("No file path specified.")
            return

        # Get napari tracks array
        self.data, self.properties, self.data_not_nan, ds = (
            self._get_data_for_layers(fps, source_software, file_path)
        )

        logger.info("Converted dataset to a napari Tracks array.")
        logger.debug(f"Tracks array shape: {self.data.shape}")

        # Set property to color points and tracks by
        self._set_common_color_property()

        # Add the data as a points and a tracks layers
        self._add_points_layer()
        self._add_tracks_layer()

        # Ensure the frame slider reflects the total number of frames
        # over all loaded layers
        self._check_frame_slider_range()

        # Set the frame slider position
        self._set_initial_state()

    def _get_data_for_layers(self, fps, source_software, file_path):
        """Load the data and convert it to a napari Tracks array."""
        # Load data as a movement dataset and convert to napari Tracks array
        loader = (
            load_poses
            if source_software in SUPPORTED_POSES_FILES
            else load_bboxes
        )
        ds = loader.from_file(file_path, source_software, fps)
        data, properties = ds_to_napari_tracks(ds)

        # Find rows that do not contain NaN values
        data_not_nan = ~np.any(np.isnan(data), axis=1)

        return (data, properties, data_not_nan, ds)

    def _set_common_color_property(self):
        """Set the color property for the Points and Tracks layers.

        The color property is set to "individual" by default,
        If the dataset contains only one individual and "keypoint"
        is defined as a property, the color property is set to "keypoint".
        """
        color_prop = "individual"
        n_individuals = len(self.properties["individual"].unique())
        if n_individuals == 1 and "keypoint" in self.properties:
            color_prop = "keypoint"
        self.color_property = color_prop

    def _set_initial_state(self):
        """Set dimension slider at first frame and points layer active."""
        # Set slider to first frame so that first view is not cluttered
        # with tracks
        default_current_step = self.viewer.dims.current_step
        self.viewer.dims.current_step = (0,) + default_current_step[2:]

        # Set last loaded points layer as active
        self.viewer.layers.selection.active = [
            ly for ly in self.viewer.layers if isinstance(ly, layers.Points)
        ][-1]

    def _add_points_layer(self):
        """Add the tracked data to the viewer as a Points layer."""
        # Define style for points layer
        points_style = PointsStyle(name=f"data: {self.file_name}")

        # Set property for markers' text
        text_prop = "individual"
        if (
            "keypoint" in self.properties
            and len(self.properties["keypoint"].unique()) > 1
        ):
            text_prop = "keypoint"
        points_style.set_text_by(property=text_prop)

        # Set color of markers and text
        points_style.set_color_by(
            property=self.color_property,
            properties_df=self.properties,
        )

        # Add data as a points layer
        points_layer = self.viewer.add_points(
            self.data[self.data_not_nan, 1:],  # (track_id), frame_idx, y, x
            properties=self.properties.iloc[self.data_not_nan, :],
            metadata={
                "max_frame_idx": max(self.data[:, 1]),
                "data": self.data,  # original data with nans
                "properties": self.properties,  # original properties with nans
                "data_not_nan": self.data_not_nan,
            },
            **points_style.as_kwargs(),
        )

        logger.info("Added tracked dataset as a napari Points layer.")

        return points_layer

    def _add_tracks_layer(self):
        """Add the tracked data to the viewer as a Tracks layer."""
        # Factorize the color property (required for tracks layer)
        codes, _ = pd.factorize(self.properties[self.color_property])
        color_property_factorized = self.color_property + "_factorized"
        self.properties[color_property_factorized] = codes

        # Define style for tracks layer
        tracks_style = TracksStyle(
            name=f"tracks: {self.file_name}",
            tail_length=int(max(self.data[:, 1])),
            # Set the tail length to the number of frames.
            # If the value is over 300, it sets the maximum
            # tail_length in the slider to the value passed.
            # It also affects the head_length slider.
        )

        # Set color by property
        tracks_style.set_color_by(property=color_property_factorized)

        # Add data as a tracks layer
        tracks_layer = self.viewer.add_tracks(
            self.data[self.data_not_nan, :],
            properties=self.properties.iloc[self.data_not_nan, :],
            **tracks_style.as_kwargs(),
        )

        # Add metadata?
        # Link to points layer?

        logger.info("Added tracked dataset as a napari Tracks layer.")

        return tracks_layer

    def _show_N_prior_markers(self, event):
        """Return callback for showing N markers before current frame.

        The event is triggered when the user changes the frame in the
        dimension slider.
        """
        list_points_layers = [
            ly for ly in self.viewer.layers if isinstance(ly, layers.Points)
        ]
        for layer in list_points_layers:
            # Get original properties dataframe with NaN values
            properties_with_nans = layer.metadata["properties"]
            data_not_nan = layer.metadata["data_not_nan"]

            # Select points that are 5 frames before the current frame
            slc_prior_frames = np.logical_and(
                properties_with_nans.loc[data_not_nan, "frame_idx"]
                > event.value[0] - 5,
                properties_with_nans.loc[data_not_nan, "frame_idx"]
                <= event.value[0],
            )

            # In the data for this layer, set the frame of all
            # selected points to current frame
            layer.data[slc_prior_frames, 0] = event.value[0]

            # Force a refresh of the points layer
            layer.refresh()

    @staticmethod
    def _enable_layer_tooltips():
        """Toggle on tooltip visibility for napari layers.

        This nicely displays the layer properties as a tooltip
        when hovering over the layer in the napari viewer.
        """
        settings = get_settings()
        settings.appearance.layer_tooltip_visibility = True
