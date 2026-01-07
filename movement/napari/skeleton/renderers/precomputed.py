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
        self.vectors = self.compute_skeleton_vectors()

    def cleanup(self) -> None:
        """Clean up resources by clearing pre-computed vectors."""
        self.vectors = None

    def compute_skeleton_vectors(self) -> np.ndarray:
        """Compute skeleton vectors in napari format.

        Returns
        -------
        np.ndarray
            Skeleton vectors in napari format.
            Shape: (N, 2, D+1) where:
            - N = total number of valid vectors
            - D = spatial dimensions (2 for 2D, 3 for 3D)
            - Each vector is [[t, y, x, ...], [dt, dy, dx, ...]]

        Notes
        -----
        The napari vectors format is crucial for correct rendering:
        - For 2D+time: shape is (N, 2, 3) with coords [t, y, x]
        - For 3D+time: shape is (N, 2, 4) with coords [t, z, y, x]
        - Vectors with NaN keypoints are skipped
        - All values are flattened into a single list of N vectors

        """
        # Get position data from dataset
        # Shape: (n_frames, n_space, n_keypoints, n_individuals)
        position = self.dataset.position.values

        # Determine if 3D based on space dimension
        is_3d = self.n_space == 3

        # List to collect all valid vectors
        vectors_list = []

        # Iterate over all frames, individuals, and connections
        for frame_idx in range(self.n_frames):
            for ind_idx in range(self.n_individuals):
                for _conn_idx, (start_kp, end_kp) in enumerate(
                    self.connections
                ):
                    # Get keypoint positions for this frame and individual
                    # Shape: (n_space,) - values are [x, y] or [x, y, z]
                    start_pos = position[frame_idx, :, start_kp, ind_idx]
                    end_pos = position[frame_idx, :, end_kp, ind_idx]

                    # Skip if either keypoint has NaN values
                    if np.any(np.isnan(start_pos)) or np.any(
                        np.isnan(end_pos)
                    ):
                        continue

                    # Convert positions to napari format
                    # napari expects [t, y, x] for 2D or [t, z, y, x] for 3D
                    # Movement stores as [x, y] or [x, y, z]
                    if is_3d:
                        # 3D: movement [x, y, z] -> napari [t, z, y, x]
                        start_napari = np.array(
                            [
                                frame_idx,
                                start_pos[2],  # z
                                start_pos[1],  # y
                                start_pos[0],  # x
                            ]
                        )
                        end_napari = np.array(
                            [
                                frame_idx,
                                end_pos[2],  # z
                                end_pos[1],  # y
                                end_pos[0],  # x
                            ]
                        )
                    else:
                        # 2D: movement [x, y] -> napari [t, y, x]
                        start_napari = np.array(
                            [
                                frame_idx,
                                start_pos[1],  # y
                                start_pos[0],  # x
                            ]
                        )
                        end_napari = np.array(
                            [
                                frame_idx,
                                end_pos[1],  # y
                                end_pos[0],  # x
                            ]
                        )

                    # Compute direction vector
                    direction = end_napari - start_napari

                    # Create vector in napari format: [start, direction]
                    vector = np.array([start_napari, direction])

                    vectors_list.append(vector)

        # Convert list to numpy array
        # Shape: (N, 2, D+1) where D is 2 or 3
        if len(vectors_list) == 0:
            # No valid vectors - return empty array with correct shape
            d_plus_1 = 4 if is_3d else 3
            return np.zeros((0, 2, d_plus_1))

        vectors = np.array(vectors_list)
        return vectors

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
