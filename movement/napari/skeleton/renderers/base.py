"""Abstract base class for skeleton renderers."""

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr


class BaseRenderer(ABC):
    """Abstract base class for skeleton renderers.

    All skeleton renderers must implement this interface to ensure
    compatibility with the skeleton visualization system.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        connections: list[tuple[int, int]],
        edge_colors: np.ndarray,
        edge_widths: np.ndarray,
    ):
        """Initialize the base renderer.

        Parameters
        ----------
        dataset : xr.Dataset
            Movement dataset containing pose data
        connections : list of tuple of int
            List of (start_keypoint_idx, end_keypoint_idx) pairs
        edge_colors : np.ndarray
            Array of RGBA colors for each connection, shape (n_edges, 4)
        edge_widths : np.ndarray
            Array of line widths for each connection, shape (n_edges,)

        """
        self.dataset = dataset
        self.connections = connections
        self.edge_colors = edge_colors
        self.edge_widths = edge_widths

        # Extract dataset dimensions
        self.n_frames = dataset.sizes["time"]
        self.n_individuals = dataset.sizes["individuals"]
        self.n_keypoints = dataset.sizes["keypoints"]
        self.n_space = dataset.sizes["space"]

        # Validate connections
        for start_idx, end_idx in connections:
            if start_idx >= self.n_keypoints or end_idx >= self.n_keypoints:
                raise ValueError(
                    f"Connection ({start_idx}, {end_idx}) references "
                    f"invalid keypoint index (max: {self.n_keypoints - 1})"
                )

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable name of this renderer."""
        ...

    @property
    @abstractmethod
    def supports_3d(self) -> bool:
        """Return whether this renderer supports 3D visualization."""
        ...

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        """Return whether this renderer requires GPU acceleration."""
        ...

    @abstractmethod
    def compute_skeleton_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute skeleton vectors in napari format.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (vectors, edge_indices) where:
            - vectors have shape (N, 2, D+1)
            - edge_indices have shape (N,)

        Notes
        -----
        The output must be in napari's expected format for vectors:

        - Shape: (N, 2, D+1) for time-series data
        - Where N is the total number of vectors (flattened across all
          frames, individuals, and connections)
        - Each vector is [[t, y, x, ...], [dt, dy, dx, ...]]
        - For 2D+time: (N, 2, 3) with positions [t, y, x]
        - For 3D+time: (N, 2, 4) with positions [t, z, y, x]

        """

    @abstractmethod
    def prepare(self) -> None:
        """Perform any setup or initialization required before rendering.

        This method is called once before the renderer is used.
        Use it to pre-compute data, allocate resources, etc.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the renderer.

        This method is called when the renderer is no longer needed.
        Use it to free memory, close connections, etc.
        """

    def update_connections(
        self,
        connections: list[tuple[int, int]],
        edge_colors: np.ndarray,
        edge_widths: np.ndarray,
    ) -> None:
        """Update the skeleton connections and re-prepare if necessary.

        Parameters
        ----------
        connections : list of tuple of int
            New list of (start_keypoint_idx, end_keypoint_idx) pairs
        edge_colors : np.ndarray
            New array of RGBA colors, shape (n_edges, 4)
        edge_widths : np.ndarray
            New array of line widths, shape (n_edges,)

        """
        self.connections = connections
        self.edge_colors = edge_colors
        self.edge_widths = edge_widths
        # Subclasses may need to re-prepare after updating connections
        self.prepare()

    def estimate_memory(self) -> float:
        """Estimate memory usage in megabytes.

        Returns
        -------
        float
            Estimated memory usage in MB

        """
        # Default implementation provides a rough estimate
        # Subclasses should override for more accurate estimates
        n_edges = len(self.connections)
        bytes_per_vector = 2 * (self.n_space + 1) * 8  # float64
        total_vectors = self.n_frames * self.n_individuals * n_edges
        return (total_vectors * bytes_per_vector) / (1024**2)

    def get_info(self) -> dict[str, str | int | float]:
        """Get renderer-specific information and metrics.

        Returns
        -------
        dict
            Dictionary with renderer information

        """
        return {
            "name": self.name,
            "supports_3d": self.supports_3d,
            "requires_gpu": self.requires_gpu,
            "n_frames": self.n_frames,
            "n_individuals": self.n_individuals,
            "n_connections": len(self.connections),
            "estimated_memory_mb": self.estimate_memory(),
        }
