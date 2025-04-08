"""Widget for converting shapes to ROI objects."""

import json
from pathlib import Path

import numpy as np
import pandas as pd  # Add this import
import h5py  # Add this import
from napari.viewer import Viewer
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from movement.roi import LineOfInterest, PolygonOfInterest
from movement.utils.logging import logger


class ShapesExporter(QWidget):
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
        self._create_export_button()
        self._create_load_button()

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
                if layer.__class__.__name__ == "Shapes":
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

    def _create_export_button(self):
        """Create a button to export ROI objects."""
        self.export_button = QPushButton("Export ROIs")
        self.export_button.setObjectName("export_button")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.layout().addRow(self.export_button)
        self.export_button.setEnabled(False)  # Disabled until ROIs are created
        self.roi_objects = None  # Store ROI objects for export

    def _create_load_button(self):
        """Create a button to load ROI objects from file."""
        self.load_button = QPushButton("Load ROIs")
        self.load_button.setObjectName("load_button")
        self.load_button.clicked.connect(self._on_load_clicked)
        self.layout().addRow(self.load_button)

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
            self._update_status(
                "Selected shapes layer contains no data", is_error=True
            )
            return

        try:
            # Convert shapes to ROI objects
            roi_objects = self._convert_shapes_to_roi(shapes_layer)

            # Create new layer for ROIs
            roi_layer = self._create_roi_layer(roi_objects)

            # Store ROI objects and enable export button
            self.roi_objects = roi_objects
            self.export_button.setEnabled(True)

            # Update status and show message
            success_msg = (
                f"""Successfully converted {len(roi_objects)}
                shapes to ROI objects\n"""
                f"Created new layer: {roi_layer.name}"
            )
            self._update_status(success_msg)
            self._show_message("Conversion Complete", success_msg)

            return roi_objects
        except Exception as e:
            error_msg = f"Error converting shapes: {str(e)}"
            self._update_status(error_msg, is_error=True)
            logger.error(error_msg)

    def _on_export_clicked(self):
        """Export ROI objects to a file."""
        if not self.roi_objects:
            self._update_status("No ROI objects to export", is_error=True)
            return

        try:
            file_filter = "JSON Files (*.json);;DeepLabCut H5 (*.h5)"
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self, "Export ROI objects", "", file_filter
            )

            if not file_path:
                return

            if selected_filter == "JSON Files (*.json)":
                if not file_path.endswith(".json"):
                    file_path += ".json"
                self._export_roi_objects(self.roi_objects, file_path)
            else:
                if not file_path.endswith(".h5"):
                    file_path += ".h5"
                self._export_roi_objects_dlc_h5(self.roi_objects, file_path)

            success_msg = f"Successfully exported ROI objects to {file_path}"
            self._update_status(success_msg)
            self._show_message("Export Complete", success_msg)

        except Exception as e:
            error_msg = f"Error exporting ROI objects: {str(e)}"
            self._update_status(error_msg, is_error=True)
            logger.error(error_msg)

    def _export_roi_objects_dlc_h5(self, roi_objects, file_path):
        """Export ROI objects to a DeepLabCut H5 file format."""
        scorer = "ROI_Export"
        bodyparts = []
        x_coords = []
        y_coords = []
        likelihood = []
        
        # Collect coordinates from ROIs
        for roi in roi_objects:
            if isinstance(roi, LineOfInterest):
                coords = np.array(roi.region.coords)
            else:  # Polygon
                coords = np.array(roi.exterior_boundary.region.coords)
            
            for j, point in enumerate(coords):
                bodyparts.append(f"{roi.name}_point_{j}")
                x_coords.append(point[0])
                y_coords.append(point[1])
                likelihood.append(1.0)
        
        # Create multi-index columns
        columns = pd.MultiIndex.from_product(
            [[scorer], bodyparts, ['x', 'y', 'likelihood']], 
            names=['scorer', 'bodyparts', 'coords']
        )
        
        # Create DataFrame with proper shape
        values = np.vstack([x_coords, y_coords, likelihood]).T
        values = values.reshape(1, -1)  # Reshape to (1, n_bodyparts * 3)
        df = pd.DataFrame(values, columns=columns)
        
        # Save to H5 file
        df.to_hdf(file_path, 'df_with_missing', format='table', mode='w')
        
        logger.info(f"Exported {len(roi_objects)} ROIs to DLC H5 format at {file_path}")

    def _on_load_clicked(self):
        """Load ROI objects from a JSON file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load ROI objects", "", "JSON Files (*.json)"
            )

            if not file_path:
                return  # User cancelled

            roi_objects = self._load_roi_objects(file_path)
            if roi_objects:
                self.roi_objects = roi_objects
                self.export_button.setEnabled(True)

                _ = self._create_roi_layer(roi_objects)

                success_msg = (
                    f"Successfully loaded {len(roi_objects)} ROI objects"
                )
                self._update_status(success_msg)
                self._show_message("Load Complete", success_msg)

        except Exception as e:
            error_msg = f"Error loading ROI objects: {str(e)}"
            self._update_status(error_msg, is_error=True)
            logger.error(error_msg)

    def _export_roi_objects(self, roi_objects, file_path):
        """Export ROI objects to a JSON file."""
        export_data = []

        for i, roi in enumerate(roi_objects):
            roi_data = {
                "id": i,
                "type": "line"
                if isinstance(roi, LineOfInterest)
                else "polygon",
                "name": getattr(roi, "name", f"roi_{i}"),
            }

            if isinstance(roi, LineOfInterest):
                roi_data["coordinates"] = roi.region.coords[:]
            else:
                roi_data["exterior_boundary"] = (
                    roi.exterior_boundary.region.coords[:]
                )
                if hasattr(roi, "holes") and roi.holes:
                    roi_data["holes"] = [
                        hole.region.coords[:] for hole in roi.holes
                    ]

            if hasattr(roi, "metadata"):
                roi_data["metadata"] = roi.metadata

            export_data.append(roi_data)

        export_data = json.loads(
            json.dumps(
                export_data,
                default=lambda x: x.tolist()
                if isinstance(x, np.ndarray)
                else x,
            )
        )

        Path(file_path).write_text(json.dumps(export_data, indent=2))

    def _load_roi_objects(self, file_path):
        """Load ROI objects from a JSON file."""
        with open(file_path) as f:
            data = json.load(f)

        roi_objects = []
        for item in data:
            try:
                if item["type"] == "line":
                    coords = np.array(item["coordinates"])
                    roi = LineOfInterest(
                        coords,
                        loop=False,
                        name=item.get(
                            "name", f"loaded_line_{len(roi_objects)}"
                        ),
                    )
                else:  # polygon
                    coords = np.array(item["exterior_boundary"])
                    holes = [np.array(h) for h in item.get("holes", [])]
                    roi = PolygonOfInterest(
                        exterior_boundary=coords,
                        holes=holes if holes else None,
                        name=item.get(
                            "name", f"loaded_polygon_{len(roi_objects)}"
                        ),
                    )

                # Restore metadata if it exists
                if "metadata" in item:
                    roi.metadata = item["metadata"]

                roi_objects.append(roi)
                logger.info(f"Loaded ROI: {roi.name}")

            except Exception as e:
                logger.warning(
                    f"""Failed to load ROI
                    {item.get("name", "unknown")}: {str(e)}"""
                )
                continue

        return roi_objects

    def _get_shapes_layer(self, layer_name):
        """Get shapes layer by name with validation."""
        try:
            layer = self.viewer.layers[layer_name]
            if layer.__class__.__name__ != "Shapes":
                logger.warning(f"Layer '{layer_name}' is not a shapes layer")
                return None
            return layer
        except KeyError:
            logger.warning(f"Could not find layer: {layer_name}")
            return None

    def _get_roi_name(self, shape_type, index, default_name):
        """Get custom name for ROI object with optional default."""
        name, ok = QInputDialog.getText(
            self,
            "Name ROI Object",
            f"""Enter name for {shape_type} {index}:""",
            text=default_name,
        )
        return name if ok and name else default_name

    def _convert_shapes_to_roi(self, shapes_layer):
        """Convert shapes to ROI objects."""
        roi_objects = []

        try:
            for i, shape in enumerate(shapes_layer.data):
                if not isinstance(shape, np.ndarray):
                    logger.warning(f"Skipping invalid shape at index {i}")
                    continue

                shape_type = (
                    shapes_layer.shape_type[i]
                    if i < len(shapes_layer.shape_type)
                    else "unknown"
                )
                coords = shape.astype(float)

                try:
                    if shape_type == "line":
                        default_name = f"line_{i}"
                        custom_name = self._get_roi_name(
                            "line", i, default_name
                        )
                        roi = LineOfInterest(
                            coords, loop=False, name=custom_name
                        )
                    elif shape_type in ["polygon", "rectangle", "ellipse"]:
                        default_name = f"{shape_type}_{i}"
                        custom_name = self._get_roi_name(
                            shape_type, i, default_name
                        )
                        roi = PolygonOfInterest(
                            exterior_boundary=coords, name=custom_name
                        )
                    else:
                        logger.warning(f"Unsupported shape type: {shape_type}")
                        continue

                    logger.info(f"Created ROI: {roi.name}")

                except Exception as e:
                    logger.warning(
                        f"Failed to create ROI for shape {i}: {str(e)}"
                    )
                    continue

                # Copy properties if they exist
                if (
                    hasattr(shapes_layer, "properties")
                    and shapes_layer.properties
                ):
                    metadata = {}
                    for (
                        prop_name,
                        prop_values,
                    ) in shapes_layer.properties.items():
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
        properties = {"name": []}

        for roi in roi_objects:
            if isinstance(roi, LineOfInterest):
                coords = np.array(roi.region.coords)
                roi_type = "line"
            else:
                coords = np.array(roi.exterior_boundary.region.coords)
                roi_type = "polygon"

            roi_data.append(coords)
            roi_types.append(roi_type)

            properties["name"].append(
                getattr(roi, "name", f"roi_{len(roi_data) - 1}")
            )

            if hasattr(roi, "metadata"):
                for key, value in roi.metadata.items():
                    if key not in properties:
                        properties[key] = []
                    properties[key].append(value)

        layer_name = self.roi_name_input.text() or "ROI Layer"
        roi_layer = self.viewer.add_shapes(
            roi_data,
            shape_type=roi_types,
            properties=properties,
            name=layer_name,
            edge_color="blue",
            face_color="transparent",
            text="{name}",
        )
        logger.info(f"Creating ROI layer with {len(roi_objects)} objects")
        return roi_layer
