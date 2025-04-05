"""Widget for converting shapes to ROI objects."""

import numpy as np
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QPushButton,
    QWidget,
    QLabel,
    QMessageBox,
    QLineEdit,
)
from qtpy.QtCore import QTimer
from movement.utils.logging import logger
from movement.roi import LineOfInterest, PolygonOfInterest  # Update imports to use correct ROI classes


class ShapesExporter(QWidget):  # Change back to ShapesExporter to match meta_widget.py
    """Widget for converting shapes from napari layers to ROI objects."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the shapes converter widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())

        # Create UI elements
        self._create_layer_selector()
        self._create_convert_button()
        self._create_status_label()
        self._create_roi_name_input()
        
        # Connect to viewer events to update layer list when layers change
        self.viewer.layers.events.inserted.connect(self._update_layer_list)
        self.viewer.layers.events.removed.connect(self._update_layer_list)
        
        # Initial update of layer list
        QTimer.singleShot(100, self._update_layer_list)

    def _create_layer_selector(self):
        """Create a combo box for selecting the shapes layer to convert."""
        self.layer_combo = QComboBox()
        self.layer_combo.setObjectName("layer_combo")
        self.layout().addRow("shapes layer:", self.layer_combo)

    def _update_layer_list(self):
        """Update the list of available shapes layers."""
        # Clear the current list
        self.layer_combo.clear()
        
        # Add all shapes layers to the combo box
        shapes_layers = []
        for layer in self.viewer.layers:
            try:
                # More robust check for shapes layers
                if layer.__class__.__name__ == 'Shapes':
                    shapes_layers.append(layer.name)
            except (AttributeError, TypeError):
                continue
                
        if shapes_layers:
            self.layer_combo.addItems(shapes_layers)
        else:
            # If no shapes layers exist, add a placeholder
            self.layer_combo.addItem("No shapes layers available")

    def _create_convert_button(self):
        """Create a button to convert the shapes data."""
        self.convert_button = QPushButton("Convert to ROI")
        self.convert_button.setObjectName("convert_button")
        self.convert_button.clicked.connect(self._on_convert_clicked)
        self.layout().addRow(self.convert_button)

    def _create_status_label(self):
        """Create a label to show conversion status."""
        self.status_label = QLabel("")
        self.layout().addRow(self.status_label)

    def _create_roi_name_input(self):
        """Create an input field for ROI layer name."""
        self.roi_name_input = QLineEdit("ROI Layer")
        self.roi_name_input.setObjectName("roi_name_input")
        self.layout().addRow("ROI layer name:", self.roi_name_input)
        
    def _show_message(self, title, text):
        """Show a message box with the given title and text."""
        QMessageBox.information(self, title, text)
        
    def _update_status(self, text, is_error=False):
        """Update the status label with text and color based on status."""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            "color: red;" if is_error else "color: green;"
        )

    def _on_convert_clicked(self):
        """Convert the shapes data to ROI objects and display them."""
        layer_name = self.layer_combo.currentText()
        
        if layer_name == "No shapes layers available":
            self._update_status("No shapes layers available", is_error=True)
            return
            
        shapes_layer = self._get_shapes_layer(layer_name)
        if shapes_layer is None:
            self._update_status("Invalid shapes layer", is_error=True)
            return
            
        if not shapes_layer.data or len(shapes_layer.data) == 0:
            self._update_status("Selected shapes layer contains no data", is_error=True)
            return
            
        try:
            # Convert shapes to ROI objects
            roi_objects = self._convert_shapes_to_roi(shapes_layer)
            
            # Create new layer for ROIs
            roi_layer = self._create_roi_layer(roi_objects)
            
            # Update status and show message
            success_msg = (
                f"Successfully converted {len(roi_objects)} shapes to ROI objects\n"
                f"Created new layer: {roi_layer.name}"
            )
            self._update_status(success_msg)
            self._show_message("Conversion Complete", success_msg)
            
            return roi_objects
        except Exception as e:
            error_msg = f"Error converting shapes: {str(e)}"
            self._update_status(error_msg, is_error=True)
            logger.error(error_msg)

    def _get_shapes_layer(self, layer_name):
        """Get shapes layer by name with validation."""
        try:
            layer = self.viewer.layers[layer_name]
            if layer.__class__.__name__ != 'Shapes':
                logger.warning(f"Layer '{layer_name}' is not a shapes layer")
                return None
            return layer
        except KeyError:
            logger.warning(f"Could not find layer: {layer_name}")
            return None
            
    def _convert_shapes_to_roi(self, shapes_layer):
        """Convert shapes to ROI objects."""
        roi_objects = []

        try:
            for i, shape in enumerate(shapes_layer.data):
                if not isinstance(shape, np.ndarray):
                    logger.warning(f"Skipping invalid shape at index {i}")
                    continue
                
                shape_type = shapes_layer.shape_type[i] if i < len(shapes_layer.shape_type) else "unknown"
                coords = shape.astype(float)  # Convert to float for ROI objects
                
                # Convert based on shape type using proper ROI classes
                try:
                    if shape_type == 'line':
                        roi = LineOfInterest(
                            coords,  # Pass coordinates directly
                            loop=False,
                            name=f"{shape_type}_{i}"
                        )
                    elif shape_type in ['polygon', 'rectangle', 'ellipse']:
                        roi = PolygonOfInterest(
                            exterior_boundary=coords,
                            name=f"{shape_type}_{i}"
                        )
                    else:
                        logger.warning(f"Unsupported shape type: {shape_type}")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to create ROI for shape {i}: {str(e)}")
                    continue
                
                # Copy properties if they exist
                if hasattr(shapes_layer, 'properties') and shapes_layer.properties:
                    metadata = {}
                    for prop_name, prop_values in shapes_layer.properties.items():
                        if i < len(prop_values):
                            metadata[prop_name] = prop_values[i]
                    roi.metadata = metadata
                
                roi_objects.append(roi)
                
            if not roi_objects:
                raise ValueError("No valid shapes to convert")
            
            return roi_objects
            
        except Exception as e:
            logger.error(f"Error converting shapes: {str(e)}")
            raise

    def _create_roi_layer(self, roi_objects):
        """Create a new shapes layer for ROI objects."""
        roi_data = []
        roi_types = []
        properties = {}
        # Added these 2 lines to check if all the shapes are converted to ROI objects
        # logger.warning(f"Converted {len(roi_objects)} shapes to ROI objects") 
        # logger.warning(f"ROI objects: {roi_objects}")
        # Convert ROI objects back to shape data
        for roi in roi_objects:
            if isinstance(roi, LineOfInterest):
                # Access coordinates through region.coords for LineOfInterest
                coords = np.array(roi.region.coords)
                roi_type = 'line'
            else:  # PolygonOfInterest
                coords = np.array(roi.exterior_boundary.region.coords)
                roi_type = 'polygon'
                
            roi_data.append(coords)
            roi_types.append(roi_type)
            
            # Copy metadata if it exists
            if hasattr(roi, 'metadata'):
                for key, value in roi.metadata.items():
                    if key not in properties:
                        properties[key] = []
                    properties[key].append(value)
        
        # Create new shapes layer for ROIs
        layer_name = self.roi_name_input.text() or "ROI Layer"
        roi_layer = self.viewer.add_shapes(
            roi_data,
            shape_type=roi_types,
            properties=properties,
            name=layer_name,
            edge_color='blue',
            face_color='transparent',
        )
        return roi_layer