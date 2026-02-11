"""Skeleton state management for Movement datasets."""

import json
from typing import Any

import numpy as np
import xarray as xr

from movement.napari.skeleton.config import (
    config_to_arrays,
    hex_to_rgba,
    rgba_to_hex,
)


class SkeletonState:
    """Manages skeleton configuration for a Movement dataset.

    This class handles the skeleton configuration including connections,
    colors, widths, and segment labels. It can save/load the configuration
    to/from the dataset's attributes or external files.

    Attributes
    ----------
    connections : list of tuple of int
        List of (start_keypoint_idx, end_keypoint_idx) pairs
    edge_colors : np.ndarray
        Array of RGBA colors for each edge, shape (n_edges, 4)
    edge_widths : np.ndarray
        Array of line widths for each edge, shape (n_edges,)
    segment_labels : list of str
        List of segment names for each edge
    keypoint_names : list of str
        List of keypoint names from the dataset

    """

    def __init__(
        self,
        connections: list[tuple[int, int]],
        edge_colors: np.ndarray,
        edge_widths: np.ndarray,
        segment_labels: list[str],
        keypoint_names: list[str],
    ):
        """Initialize skeleton state.

        Parameters
        ----------
        connections : list of tuple of int
            List of (start_idx, end_idx) pairs
        edge_colors : np.ndarray
            RGBA colors array, shape (n_edges, 4)
        edge_widths : np.ndarray
            Line widths array, shape (n_edges,)
        segment_labels : list of str
            Segment labels for each edge
        keypoint_names : list of str
            Keypoint names from the dataset

        """
        self.connections = connections
        self.edge_colors = edge_colors
        self.edge_widths = edge_widths
        self.segment_labels = segment_labels
        self.keypoint_names = keypoint_names

    @classmethod
    def from_config(
        cls, config: dict[str, Any], dataset: xr.Dataset
    ) -> "SkeletonState":
        """Create SkeletonState from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Skeleton configuration dictionary
        dataset : xr.Dataset
            Movement dataset

        Returns
        -------
        SkeletonState
            New SkeletonState instance

        """
        keypoint_names = list(dataset.coords["keypoints"].values)
        connections, colors, widths, labels = config_to_arrays(
            config, keypoint_names
        )

        return cls(
            connections=connections,
            edge_colors=colors,
            edge_widths=widths,
            segment_labels=labels,
            keypoint_names=keypoint_names,
        )

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> "SkeletonState | None":
        """Extract skeleton state from dataset attributes.

        Parameters
        ----------
        dataset : xr.Dataset
            Movement dataset with skeleton configuration in attributes

        Returns
        -------
        SkeletonState or None
            SkeletonState instance if configuration exists, None otherwise

        """
        # Check if skeleton config exists in dataset attributes
        if "skeleton_config" not in dataset.attrs:
            return None

        # Parse the skeleton config JSON
        config_json = dataset.attrs["skeleton_config"]
        config = json.loads(config_json)

        return cls.from_config(config, dataset)

    def to_dataset_attrs(self, dataset: xr.Dataset) -> xr.Dataset:
        """Embed skeleton configuration into dataset attributes.

        Parameters
        ----------
        dataset : xr.Dataset
            Movement dataset to add skeleton configuration to

        Returns
        -------
        xr.Dataset
            Dataset with skeleton configuration in attributes

        """
        # Convert state to config dictionary
        config = self.to_config_dict()

        # Store as JSON in dataset attributes
        dataset.attrs["skeleton_config"] = json.dumps(config)
        dataset.attrs["skeleton_config_version"] = "1.0"

        return dataset

    def to_config_dict(self) -> dict[str, Any]:
        """Convert skeleton state to configuration dictionary.

        Returns
        -------
        dict
            Configuration dictionary suitable for YAML export

        """
        connections_list = []
        for i, (start_idx, end_idx) in enumerate(self.connections):
            conn_dict = {
                "start": self.keypoint_names[start_idx],
                "end": self.keypoint_names[end_idx],
                "color": rgba_to_hex(self.edge_colors[i]),
                "width": float(self.edge_widths[i]),
                "segment": self.segment_labels[i],
            }
            connections_list.append(conn_dict)

        return {
            "keypoints": self.keypoint_names,
            "connections": connections_list,
        }

    def add_connection(
        self,
        start_keypoint: str,
        end_keypoint: str,
        color: str = "#FFFFFF",
        width: float = 2.0,
        segment: str = "",
    ) -> None:
        """Add a new connection to the skeleton.

        Parameters
        ----------
        start_keypoint : str
            Name of the starting keypoint
        end_keypoint : str
            Name of the ending keypoint
        color : str, optional
            Hex color string, by default "#FFFFFF"
        width : float, optional
            Line width, by default 2.0
        segment : str, optional
            Segment label, by default ""

        Raises
        ------
        ValueError
            If keypoint names are not found in the dataset

        """
        if start_keypoint not in self.keypoint_names:
            raise ValueError(
                f"Start keypoint '{start_keypoint}' not found "
                f"in dataset keypoints"
            )
        if end_keypoint not in self.keypoint_names:
            raise ValueError(
                f"End keypoint '{end_keypoint}' not found in dataset keypoints"
            )

        start_idx = self.keypoint_names.index(start_keypoint)
        end_idx = self.keypoint_names.index(end_keypoint)

        # Add connection
        self.connections.append((start_idx, end_idx))

        # Add color
        rgba = hex_to_rgba(color)
        self.edge_colors = np.vstack([self.edge_colors, rgba])

        # Add width
        self.edge_widths = np.append(self.edge_widths, width)

        # Add segment label
        self.segment_labels.append(segment)

    def remove_connection(
        self, start_keypoint: str, end_keypoint: str
    ) -> bool:
        """Remove a connection from the skeleton.

        Parameters
        ----------
        start_keypoint : str
            Name of the starting keypoint
        end_keypoint : str
            Name of the ending keypoint

        Returns
        -------
        bool
            True if connection was removed, False if not found

        """
        if start_keypoint not in self.keypoint_names:
            return False
        if end_keypoint not in self.keypoint_names:
            return False

        start_idx = self.keypoint_names.index(start_keypoint)
        end_idx = self.keypoint_names.index(end_keypoint)

        # Find the connection
        try:
            conn_idx = self.connections.index((start_idx, end_idx))
        except ValueError:
            return False

        # Remove connection
        self.connections.pop(conn_idx)

        # Remove corresponding color, width, and label
        self.edge_colors = np.delete(self.edge_colors, conn_idx, axis=0)
        self.edge_widths = np.delete(self.edge_widths, conn_idx)
        self.segment_labels.pop(conn_idx)

        return True

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the skeleton configuration.

        Returns
        -------
        tuple of (bool, list of str)
            (is_valid, error_messages)
            If valid, error_messages will be empty

        """
        errors = []

        # Check that all arrays have consistent lengths
        n_connections = len(self.connections)
        if len(self.edge_colors) != n_connections:
            errors.append(
                f"edge_colors has {len(self.edge_colors)} entries, "
                f"expected {n_connections}"
            )
        if len(self.edge_widths) != n_connections:
            errors.append(
                f"edge_widths has {len(self.edge_widths)} entries, "
                f"expected {n_connections}"
            )
        if len(self.segment_labels) != n_connections:
            errors.append(
                f"segment_labels has {len(self.segment_labels)} entries, "
                f"expected {n_connections}"
            )

        # Check that all connection indices are valid
        max_idx = len(self.keypoint_names) - 1
        for i, (start_idx, end_idx) in enumerate(self.connections):
            if start_idx > max_idx:
                errors.append(
                    f"Connection {i}: start index {start_idx} exceeds "
                    f"max keypoint index {max_idx}"
                )
            if end_idx > max_idx:
                errors.append(
                    f"Connection {i}: end index {end_idx} exceeds "
                    f"max keypoint index {max_idx}"
                )

        is_valid = len(errors) == 0
        return is_valid, errors
