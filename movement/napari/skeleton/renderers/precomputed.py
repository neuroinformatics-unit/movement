"""Precomputed skeleton renderer for Movement napari plugin."""

import numpy as np
import xarray as xr

from movement.napari.skeleton.renderers.base import BaseRenderer


class PrecomputedRenderer(BaseRenderer):
    """Precomputed skeleton renderer.

    This renderer pre-computes all skeleton vectors for all frames upfront
    and stores them in memory. It provides the smoothest playback at the
    cost of higher memory usage.

    Best for: Small datasets (< 5,000 frames, < 20 keypoints)

    The renderer flattens all skeleton vectors across frames, individuals,
    and connections into a single numpy array in napari's expected format:
    (N, 2, D+1) where N is the total number of vectors.

    Attributes
    ----------
    vectors : np.ndarray or None
        Pre-computed skeleton vectors in napari format, shape (N, 2, D+1).
        None until prepare() is called.

    """

    def __init__(
        self,
        dataset: xr.Dataset,
        connections: list[tuple[int, int]],
        edge_colors: np.ndarray,
        edge_widths: np.ndarray,
    ):
        """Initialize the precomputed renderer.

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
        super().__init__(dataset, connections, edge_colors, edge_widths)
        self.vectors: np.ndarray | None = None
        self.edge_indices: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Return the name of this renderer."""
        return "Precomputed"

    @property
    def supports_3d(self) -> bool:
        """Return whether this renderer supports 3D visualization."""
        return True

    @property
    def requires_gpu(self) -> bool:
        """Return whether this renderer requires GPU acceleration."""
        return False

    def prepare(self) -> None:
        """Pre-compute all skeleton vectors.

        This method computes skeleton vectors for all frames, individuals,
        and connections, and stores them in the napari-expected format.
        """
        self.vectors, self.edge_indices = self.compute_skeleton_vectors()

    def cleanup(self) -> None:
        """Clean up resources by clearing pre-computed vectors."""
        self.vectors = None
        self.edge_indices = None

    def _convert_position_to_napari(
        self, pos: np.ndarray, frame_idx: int, is_3d: bool
    ) -> np.ndarray:
        """Convert position from Movement to napari format.

        Parameters
        ----------
        pos : np.ndarray
            Position in Movement format [x, y] or [x, y, z]
        frame_idx : int
            Frame index
        is_3d : bool
            Whether this is 3D data

        Returns
        -------
        np.ndarray
            Position in napari format [t, y, x] or [t, z, y, x]

        """
        if is_3d:
            # 3D: movement [x, y, z] -> napari [t, z, y, x]
            return np.array([frame_idx, pos[2], pos[1], pos[0]])
        # 2D: movement [x, y] -> napari [t, y, x]
        return np.array([frame_idx, pos[1], pos[0]])

    def _create_vector_from_positions(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        frame_idx: int,
        is_3d: bool,
    ) -> np.ndarray | None:
        """Create a skeleton vector from two keypoint positions.

        Parameters
        ----------
        start_pos : np.ndarray
            Start keypoint position
        end_pos : np.ndarray
            End keypoint position
        frame_idx : int
            Frame index
        is_3d : bool
            Whether this is 3D data

        Returns
        -------
        np.ndarray or None
            Vector in napari format [start, direction], or None if invalid

        """
        # Skip if either keypoint has NaN values
        if np.any(np.isnan(start_pos)) or np.any(np.isnan(end_pos)):
            return None

        # Convert positions to napari format
        start_napari = self._convert_position_to_napari(
            start_pos, frame_idx, is_3d
        )
        end_napari = self._convert_position_to_napari(
            end_pos, frame_idx, is_3d
        )

        # Compute direction vector and create vector in napari format
        direction = end_napari - start_napari
        return np.array([start_napari, direction])

    def compute_skeleton_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute skeleton vectors in napari format.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (vectors, edge_indices) where:
            - vectors: Skeleton vectors in napari format, shape (N, 2, D+1)
            - edge_indices: Index of the connection for each vector, shape (N,)

        Notes
        -----
        The napari vectors format is crucial for correct rendering:
        - For 2D+time: shape is (N, 2, 3) with coords [t, y, x]
        - For 3D+time: shape is (N, 2, 4) with coords [t, z, y, x]
        - Vectors with NaN keypoints are skipped
        - All values are flattened into a single list of N vectors
        - edge_indices tracks which connection each vector corresponds to

        """
        # Get position data from dataset
        position = self.dataset.position.values
        is_3d = self.n_space == 3
        vectors_list = []
        edge_indices_list = []

        # Iterate over all frames, individuals, and connections
        for frame_idx in range(self.n_frames):
            for ind_idx in range(self.n_individuals):
                for conn_idx, (start_kp, end_kp) in enumerate(
                    self.connections
                ):
                    # Get keypoint positions
                    start_pos = position[frame_idx, :, start_kp, ind_idx]
                    end_pos = position[frame_idx, :, end_kp, ind_idx]

                    # Create vector if positions are valid
                    vector = self._create_vector_from_positions(
                        start_pos, end_pos, frame_idx, is_3d
                    )
                    if vector is not None:
                        vectors_list.append(vector)
                        edge_indices_list.append(conn_idx)

        # Convert lists to numpy arrays
        if len(vectors_list) == 0:
            # No valid vectors - return empty arrays with correct shape
            d_plus_1 = 4 if is_3d else 3
            return np.zeros((0, 2, d_plus_1)), np.array([], dtype=int)

        return np.array(vectors_list), np.array(edge_indices_list)

    def estimate_memory(self) -> float:
        """Estimate memory usage in megabytes.

        Returns
        -------
        float
            Estimated memory usage in MB

        """
        # Each vector: (2, D+1) where D is spatial dims
        # D+1 is 3 for 2D+time, 4 for 3D+time
        d_plus_1 = 4 if self.n_space == 3 else 3
        bytes_per_vector = 2 * d_plus_1 * 8  # float64
        n_edges = len(self.connections)
        # Assume not all vectors are valid (some have NaN)
        # Use 80% as estimate
        estimated_valid_vectors = (
            self.n_frames * self.n_individuals * n_edges * 0.8
        )
        return (estimated_valid_vectors * bytes_per_vector) / (1024**2)

    def get_info(self) -> dict[str, str | int | float]:
        """Get renderer-specific information and metrics.

        Returns
        -------
        dict
            Dictionary with renderer information including number of
            pre-computed vectors if available.

        """
        info = super().get_info()
        if self.vectors is not None:
            info["n_precomputed_vectors"] = len(self.vectors)
            info["actual_memory_mb"] = self.vectors.nbytes / (1024**2)
        return info
