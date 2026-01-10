"""Widget for displaying color-to-keypoint legend in napari."""

from typing import Any

import numpy as np
import pandas as pd
from napari.layers import Points
from napari.utils.colormaps import ensure_colormap
from napari.viewer import Viewer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from movement.utils.logging import logger


class LegendWidget(QWidget):
    """Widget displaying color-to-keypoint/individual mapping legend."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the legend widget.

        Parameters
        ----------
        napari_viewer : napari.viewer.Viewer
            The napari viewer instance.
        parent : QWidget, optional
            Parent widget, by default None

        """
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.current_layer = None

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create widgets
        self._create_layer_selector()
        self._create_legend_list()
        self._create_controls()

        # Connect to layer events
        self._connect_layer_events()

        # Initial update
        self._update_legend()

    def _create_layer_selector(self):
        """Create a label showing which layer the legend is based on."""
        self.layer_label = QLabel("No movement layers found")
        self.layer_label.setWordWrap(True)
        self.layout().addWidget(self.layer_label)

    def _create_legend_list(self):
        """Create the list widget to display the legend items."""
        self.legend_list = QListWidget()
        self.legend_list.setMaximumHeight(200)
        self.legend_list.setAlternatingRowColors(True)
        self.layout().addWidget(self.legend_list)

    def _create_controls(self):
        """Create control buttons."""
        controls_layout = QHBoxLayout()

        # Auto-update checkbox
        self.auto_update_checkbox = QCheckBox("Auto-update")
        self.auto_update_checkbox.setChecked(True)
        self.auto_update_checkbox.setToolTip(
            "Automatically update legend when layers change"
        )
        controls_layout.addWidget(self.auto_update_checkbox)

        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Manually refresh the legend")
        self.refresh_button.clicked.connect(self._update_legend)
        controls_layout.addWidget(self.refresh_button)

        self.layout().addLayout(controls_layout)

    def _connect_layer_events(self):
        """Connect to napari layer events to auto-update legend."""
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self.viewer.layers.events.changed.connect(self._on_layer_change)

    def _on_layer_change(self, event=None):
        """Handle layer change events."""
        if self.auto_update_checkbox.isChecked():
            # Use QTimer to delay update slightly,
            # allowing layer to fully initialize
            try:
                from qtpy.QtCore import QTimer

                # Single-shot timer with 100ms delay
                QTimer.singleShot(100, self._update_legend)
            except Exception:
                # Fallback: update immediately if QTimer fails
                try:
                    self._update_legend()
                except Exception as e:
                    logger.debug(f"Error updating legend: {e}")

    def _find_movement_points_layers(self) -> list[Points]:
        """Find all Points layers that appear to be from movement.

        Parameters
        ----------
        list of Points
            List of Points layers that likely contain movement data.

        """
        movement_layers = []
        for layer in self.viewer.layers:
            if isinstance(layer, Points) and (
                hasattr(layer, "properties") and layer.properties is not None
            ):
                props = layer.properties
                # Napari stores properties as dict (keys are property names)
                # Check for movement-specific properties
                if (
                    isinstance(props, dict)
                    and (
                        "individual" in props
                        or "keypoint" in props
                        or "confidence" in props
                    )
                    or isinstance(props, pd.DataFrame)
                    and (
                        # Properties is a DataFrame
                        # (shouldn't happen in napari, but handle it)
                        "individual" in props.columns
                        or "keypoint" in props.columns
                        or "confidence" in props.columns
                    )
                ):
                    movement_layers.append(layer)
        return movement_layers

    def _get_color_mapping_from_layer(  # noqa: C901
        self, layer: Points
    ) -> dict[str, Any]:
        """Extract color mapping from a napari Points layer.

        Parameters
        ----------
        layer : napari.layers.Points
            The napari Points layer.

        Returns
        -------
        dict
            Dictionary with keys:
            - "mapping": dict mapping keypoint/individual names to RGB tuples
            - "property": str, property used for coloring (e.g., "keypoint")
            - "colormap": str, the colormap name used

        """
        if not hasattr(layer, "properties") or layer.properties is None:
            return {}

        properties = layer.properties

        # Napari stores properties as a dict, not DataFrame
        # Convert to DataFrame for easier handling if needed
        if isinstance(properties, dict):
            # Check if dict is empty
            if not properties:
                return {}
            # Convert dict to DataFrame for easier manipulation
            try:
                properties_df = pd.DataFrame(properties)
            except Exception as e:
                logger.debug(
                    f"Error converting properties dict to DataFrame: {e}"
                )
                return {}
        elif isinstance(properties, pd.DataFrame):
            properties_df = properties
            if properties_df.empty:
                return {}
        else:
            return {}

        # Determine what property is used for coloring
        color_property = None
        face_color = layer.face_color

        # Check if face_color is a string (property name) or a colormap
        if isinstance(face_color, str):
            # Color by property name
            if face_color in properties_df.columns:
                color_property = face_color
            # Might be a colormap name, try to infer from properties
            elif face_color in ["turbo", "viridis", "plasma", "inferno"]:
                # This is a colormap name, try to infer property
                if (
                    "keypoint" in properties_df.columns
                    and len(properties_df["keypoint"].unique()) > 1
                ):
                    color_property = "keypoint"
                elif (
                    "individual" in properties_df.columns
                    and len(properties_df["individual"].unique()) > 1
                ):
                    color_property = "individual"
        else:
            # Single color or array - try to infer from properties
            if (
                "keypoint" in properties_df.columns
                and len(properties_df["keypoint"].unique()) > 1
            ):
                color_property = "keypoint"
            elif (
                "individual" in properties_df.columns
                and len(properties_df["individual"].unique()) > 1
            ):
                color_property = "individual"

        if (
            color_property is None
            or color_property not in properties_df.columns
        ):
            return {}

        # Get unique values (sorted for consistent ordering)
        unique_values = sorted(properties_df[color_property].unique())
        n_colors = len(unique_values)

        # Get colormap name - try multiple ways
        from movement.napari.layer_styles import DEFAULT_COLORMAP

        colormap_name = DEFAULT_COLORMAP

        # Try to get from layer's face_colormap
        if hasattr(layer, "face_colormap") and layer.face_colormap is not None:
            try:
                colormap_obj = layer.face_colormap
                if isinstance(colormap_obj, str):
                    colormap_name = colormap_obj
                elif hasattr(colormap_obj, "name"):
                    colormap_name = colormap_obj.name
                else:
                    # Try to get from layer metadata (if dict) or use default
                    if hasattr(layer, "metadata") and isinstance(
                        layer.metadata, dict
                    ):
                        colormap_name = layer.metadata.get(
                            "colormap", DEFAULT_COLORMAP
                        )
            except Exception as e:
                logger.debug(f"Could not get colormap from layer: {e}")

        # Reconstruct color cycle using same logic as
        # PointsStyle.set_color_by()
        # This ensures consistency with how colors were originally assigned
        try:
            from movement.napari.layer_styles import _sample_colormap

            color_cycle = _sample_colormap(n_colors, colormap_name)
        except (ImportError, AttributeError, Exception):
            # Fallback: generate colors directly using the same approach
            try:
                cmap = ensure_colormap(colormap_name)
                samples = np.linspace(
                    0, len(cmap.colors) - 1, n_colors
                ).astype(int)
                color_cycle = [tuple(cmap.colors[i]) for i in samples]
            except Exception as e:
                logger.debug(f"Could not generate color cycle: {e}")
                # Last resort: use default colormap
                cmap = ensure_colormap(DEFAULT_COLORMAP)
                samples = np.linspace(
                    0, len(cmap.colors) - 1, n_colors
                ).astype(int)
                color_cycle = [tuple(cmap.colors[i]) for i in samples]

        # Create mapping (preserve order)
        mapping = {}
        for i, value in enumerate(unique_values):
            if i < len(color_cycle):
                mapping[str(value)] = color_cycle[i]

        return {
            "mapping": mapping,
            "property": color_property,
            "colormap": colormap_name,
        }

    def _update_legend(self):
        """Update the legend display based on current layers."""
        try:
            # Clear current legend
            self.legend_list.clear()

            # Find movement points layers
            movement_layers = self._find_movement_points_layers()

            if not movement_layers:
                self.layer_label.setText("No movement layers found")
                self.current_layer = None
                return

            # Use the first movement layer, or the currently selected one
            # if it's a movement layer
            selected_layers = self.viewer.layers.selection
            selected_movement_layer = None
            if selected_layers:
                for layer in selected_layers:
                    if layer in movement_layers:
                        selected_movement_layer = layer
                        break

            layer_to_use = selected_movement_layer or movement_layers[0]
            self.current_layer = layer_to_use

            # Get color mapping
            color_info = self._get_color_mapping_from_layer(layer_to_use)

            if not color_info or not color_info.get("mapping"):
                self.layer_label.setText(
                    f"Layer: {layer_to_use.name}\n"
                    "No color mapping found (layer may use single color)"
                )
                return

            # Update layer label
            property_name = color_info["property"]
            colormap_name = color_info["colormap"]
            self.layer_label.setText(
                f"Layer: {layer_to_use.name}\n"
                f"Colored by: {property_name} | Colormap: {colormap_name}"
            )

            # Populate legend list
            mapping = color_info["mapping"]
            for name, color in sorted(mapping.items()):
                item = QListWidgetItem()
                item.setText(name)

                # Set background color as a small square icon
                # Convert color to QColor (handle both 0-1 and 0-255 ranges)
                rgb = tuple(
                    int(c * 255) if c <= 1.0 else int(c) for c in color[:3]
                )
                qcolor = QColor(*rgb)

                # Create a colored icon/background
                item.setBackground(qcolor)

                # Set text color based on background brightness
                brightness = (
                    0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                )  # Perceived brightness
                text_color = (
                    QColor(255, 255, 255)
                    if brightness < 128
                    else QColor(0, 0, 0)
                )
                item.setForeground(text_color)

                # Add tooltip with RGB values
                item.setToolTip(f"{name}\nRGB: {rgb}")

                self.legend_list.addItem(item)

            logger.debug(
                f"Updated legend with {len(mapping)} entries "
                f"from layer {layer_to_use.name}"
            )
        except Exception as e:
            logger.debug(f"Error updating legend: {e}")
            self.layer_label.setText(f"Error updating legend: {str(e)}")
