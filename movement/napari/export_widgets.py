"""Widgets for exporting shapes from napari layers."""

import numpy as np
import pandas as pd
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QPushButton,
    QWidget,
)
from qtpy.QtCore import QTimer

from movement.utils.logging import logger


class ShapesExporter(QWidget):
    """Widget for exporting shapes from napari layers to files."""

    def __init__(self, napari_viewer: Viewer, parent=None):
        """Initialize the shapes exporter widget."""
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())

        # Create UI elements
        self._create_layer_selector()
        self._create_export_button()
        
        # Connect to viewer events to update layer list when layers change
        self.viewer.layers.events.inserted.connect(self._update_layer_list)
        self.viewer.layers.events.removed.connect(self._update_layer_list)
        
        # Initial update of layer list
        QTimer.singleShot(100, self._update_layer_list)

    def _create_layer_selector(self):
        """Create a combo box for selecting the shapes layer to export."""
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

    def _create_export_button(self):
        """Create a button to export the shapes data."""
        self.export_button = QPushButton("Export Shapes")
        self.export_button.setObjectName("export_button")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.layout().addRow(self.export_button)

    def _on_export_clicked(self):
        """Export the shapes data to a file when the button is clicked."""
        # Get the selected layer name
        layer_name = self.layer_combo.currentText()
        
        # Check if there are any shapes layers
        if layer_name == "No shapes layers available":
            logger.warning("No shapes layers available to export")
            return
            
        # Get the shapes layer by name
        shapes_layer = self._get_shapes_layer(layer_name)
        if shapes_layer is None:
            return
            
        # Validate shapes data
        if not shapes_layer.data or len(shapes_layer.data) == 0:
            logger.warning("Selected shapes layer contains no data")
            return
            
        # Get a file path for saving
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save shapes data",
            "",
            "CSV Files (*.csv);;JSON Files (*.json)",
        )
        
        if not file_path:
            return  # User cancelled
            
        # Ensure proper file extension
        if selected_filter == "CSV Files (*.csv)" and not file_path.endswith('.csv'):
            file_path += '.csv'
        elif selected_filter == "JSON Files (*.json)" and not file_path.endswith('.json'):
            file_path += '.json'
            
        # Export the shapes data
        try:
            self._export_shapes(shapes_layer, file_path)
        except Exception as e:
            logger.error(f"Error exporting shapes: {str(e)}")

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
            
    def _export_shapes(self, shapes_layer, file_path):
        """Export the shapes data to the specified file path."""
        shape_data = []
        
        try:
            # Get all properties from the layer
            properties = shapes_layer.properties or {}
            
            # Process each shape
            for i, shape in enumerate(shapes_layer.data):
                if not isinstance(shape, np.ndarray):
                    logger.warning(f"Skipping invalid shape at index {i}")
                    continue
                    
                # Get shape properties
                shape_type = shapes_layer.shape_type[i] if i < len(shapes_layer.shape_type) else "unknown"
                
                # For each shape, create a row for each vertex
                for j, vertex in enumerate(shape):
                    row_data = {
                        'shape_id': i,
                        'shape_type': shape_type,
                        'vertex_id': j,
                    }
                    
                    # Add coordinates
                    vertex_array = np.asarray(vertex).flatten()
                    coordinates = ['y', 'x', 'z']
                    for k, coord in enumerate(coordinates):
                        row_data[coord] = vertex_array[k] if k < len(vertex_array) else 0
                    
                    # Add any additional properties
                    for prop_name, prop_values in properties.items():
                        if i < len(prop_values):
                            row_data[prop_name] = prop_values[i]
                    
                    shape_data.append(row_data)
                    
            if not shape_data:
                raise ValueError("No valid shapes data to export")
                
            # Convert to dataframe
            df = pd.DataFrame(shape_data)
            
            # Save to file based on extension
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
            elif file_path.endswith('.json'):
                df.to_json(file_path, orient='records')
            
            logger.info(f"Successfully exported {len(shape_data)} shape vertices to {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing shapes: {str(e)}")
            raise