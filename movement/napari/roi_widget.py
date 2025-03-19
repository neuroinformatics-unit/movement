"""Widget for drawing and managing regions of interest (ROIs) in napari."""

import logging
from typing import cast

import numpy as np
from napari.layers import Shapes
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

class ROIDrawingWidget(QWidget):
    """Widget for drawing and managing regions of interest in napari."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the ROI drawing widget.

        Parameters
        ----------
        napari_viewer : napari.viewer.Viewer
            The napari viewer instance to use.
        parent : QWidget, optional
            Parent widget, by default None.

        """
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.roi_layer: Shapes | None = None
        self._selected_roi: int | None = None
        self._setup_ui()

    def _setup_ui(self):
        """Create and arrange the widget UI elements."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Buttons layout
        button_layout = QHBoxLayout()

        # Create ROI layer button
        self.create_layer_btn = QPushButton("Create ROI Layer")
        self.create_layer_btn.clicked.connect(self._create_roi_layer)
        button_layout.addWidget(self.create_layer_btn)

        # Clear ROIs button
        self.clear_btn = QPushButton("Clear ROIs")
        self.clear_btn.clicked.connect(self._clear_rois)
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def _create_roi_layer(self):
        """Create a new shapes layer for ROIs if it doesn't exist."""
        if self.roi_layer is None:
            self.roi_layer = cast(
                Shapes,
                self.viewer.add_shapes(
                    name="ROIs",
                    edge_width=2,
                    edge_color="red",
                    face_color="transparent",
                ),
            )
            self.roi_layer.mode = "add_rectangle"

            # Connect modern event handlers
            self.roi_layer.mouse_drag_callbacks.append(self._on_mouse_press)
            self.roi_layer.events.data.connect(self._on_data_change)

            self.status_label.setText(
                "ROI layer created. Left-click to draw, Right-click to select!"
            )
        else:
            self.status_label.setText("ROI layer already exists")

    def _on_mouse_press(self, layer: Shapes, event):
        """Handle mouse press events for selection and drawing."""
        if event.button == 2 and self.roi_layer is not None:  # Right click
            # Convert mouse position to data coordinates
            pos = np.array(event.position)

            # Find which ROI contains the click position
            self._selected_roi = None
            for idx, roi in enumerate(layer.data):
                if self._point_in_rectangle(pos, roi):
                    self._selected_roi = idx
                    break

            if self._selected_roi is not None:
                layer.selected_data = {self._selected_roi}
                self.status_label.setText(
                    f"Selected ROI {self._selected_roi + 1}"
                )
            else:
                layer.selected_data = set()
                self.status_label.setText("No ROI selected")

            event.handled = True

    @staticmethod
    def _point_in_rectangle(point: np.ndarray, rectangle: np.ndarray) -> bool:
        """Check if a point is inside a rectangle.
        
        Parameters
        ----------
        point : np.ndarray
            The (x, y) coordinates to check
        rectangle : np.ndarray
            Array of rectangle vertices
            
        Returns
        -------
        bool
            True if point is inside the rectangle, False otherwise

        """
        min_coords = np.min(rectangle, axis=0)
        max_coords = np.max(rectangle, axis=0)
        return bool(
            np.all(point >= min_coords) and np.all(point <= max_coords)
        )

    def _clear_rois(self):
        """Clear all ROIs from the layer."""
        if self.roi_layer is not None:
            self.roi_layer.data = []
            self._selected_roi = None
            self.status_label.setText("All ROIs cleared")

    def _on_data_change(self, event):
        """Update ROI count display."""
        if self.roi_layer is not None:
            n_rois = len(self.roi_layer.data)
            self.status_label.setText(f"Total ROIs: {n_rois}")

    def get_rois(self) -> list[np.ndarray]:
        """Get the list of ROIs as numpy arrays.
        
        Returns
        -------
        list[np.ndarray]
            List of ROI coordinates

        """
        return list(self.roi_layer.data) if self.roi_layer else []

    def get_selected_roi(self) -> np.ndarray | None:
        """Get the currently selected ROI.
        
        Returns
        -------
        np.ndarray | None
            Coordinates of the selected ROI or None if none selected

        """
        if self.roi_layer is not None and self._selected_roi is not None:
            return self.roi_layer.data[self._selected_roi]
        return None