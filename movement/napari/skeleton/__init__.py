"""Skeleton visualization for Movement napari plugin.

This module provides functionality to visualize animal skeletons
as line connections between tracked keypoints in napari.
"""

from typing import Any

import napari
import numpy as np
import xarray as xr

from movement.napari.skeleton.config import (
    config_to_arrays,
    load_yaml_config,
    save_yaml_config,
    validate_config,
)
from movement.napari.skeleton.renderers import PrecomputedRenderer
from movement.napari.skeleton.state import SkeletonState
from movement.napari.skeleton.templates import TEMPLATES

__all__ = [
    "add_skeleton_layer",
    "load_skeleton_config",
    "save_skeleton_config",
    "TEMPLATES",
    "SkeletonState",
    "PrecomputedRenderer",
]


def _validate_dataset(dataset: xr.Dataset) -> None:
    """Validate that dataset is suitable for skeleton visualization."""
    if dataset.ds_type != "poses":
        raise ValueError(
            "Dataset must be a poses dataset with 'position' and "
            "'keypoints' data"
        )
    if "keypoints" not in dataset.coords:
        raise ValueError("Dataset must have 'keypoints' coordinate")


def _resolve_skeleton_state(
    connections: dict[str, Any] | str | None, dataset: xr.Dataset
) -> SkeletonState:
    """Resolve skeleton configuration to SkeletonState.

    Parameters
    ----------
    connections : dict, str, or None
        Skeleton configuration
    dataset : xr.Dataset
        Movement dataset

    Returns
    -------
    SkeletonState
        Resolved skeleton state

    Raises
    ------
    ValueError
        If configuration is invalid
    KeyError
        If template not found
    """
    if connections is None:
        # Try to load from dataset
        skeleton_state = SkeletonState.from_dataset(dataset)
        if skeleton_state is None:
            raise ValueError(
                "No skeleton configuration found. "
                "Provide 'connections' parameter or use a dataset "
                "with embedded skeleton configuration."
            )
        return skeleton_state

    if isinstance(connections, str):
        # Load template by name
        if connections not in TEMPLATES:
            available = ", ".join(TEMPLATES.keys())
            raise KeyError(
                f"Template '{connections}' not found. "
                f"Available templates: {available}"
            )
        config: dict[str, Any] = dict(TEMPLATES[connections])
        return SkeletonState.from_config(config, dataset)

    if isinstance(connections, dict):
        # Use provided configuration
        is_valid, errors = validate_config(connections, dataset)
        if not is_valid:
            error_msg = "\n".join(errors)
            raise ValueError(f"Invalid skeleton configuration:\n{error_msg}")
        return SkeletonState.from_config(connections, dataset)

    raise ValueError(
        "connections must be a dict, str (template name), or None"
    )


def _build_vector_colors(
    skeleton_state: SkeletonState, edge_indices: np.ndarray
) -> np.ndarray:
    """Build color array for vectors using edge indices.

    Parameters
    ----------
    skeleton_state : SkeletonState
        Skeleton state with edge colors
    edge_indices : np.ndarray
        Array of edge indices for each vector, shape (N,)

    Returns
    -------
    np.ndarray
        Color array matching vectors, shape (N, 4)

    Notes
    -----
    This function uses the edge_indices returned by the renderer to ensure
    that colors are correctly mapped to vectors, even when NaN values cause
    some connections to be skipped.
    """
    # Build colors by looking up the color for each edge index
    vector_colors = np.array(
        [skeleton_state.edge_colors[idx] for idx in edge_indices]
    )
    return vector_colors


def add_skeleton_layer(
    viewer: napari.Viewer,
    dataset: xr.Dataset,
    connections: dict[str, Any] | str | None = None,
    name: str = "skeleton",
    **kwargs,
) -> napari.layers.Vectors:
    """Add a skeleton visualization layer to the napari viewer.

    This function creates a skeleton visualization by connecting keypoints
    with lines. The skeleton is rendered using napari's Vectors layer,
    which ensures smooth playback and proper line rendering.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance to add the layer to
    dataset : xr.Dataset
        Movement dataset containing pose data with 'position' and
        'keypoints' coordinates
    connections : dict, str, or None, optional
        Skeleton connection configuration. Can be:
        - dict: Configuration dictionary with 'connections' key
        - str: Name of a template from TEMPLATES (e.g., "mouse", "rat")
        - None: Attempt to load from dataset attributes
    name : str, optional
        Name for the skeleton layer, by default "skeleton"
    **kwargs
        Additional keyword arguments passed to viewer.add_vectors()

    Returns
    -------
    napari.layers.Vectors
        The created skeleton vectors layer

    Raises
    ------
    ValueError
        If dataset is not a valid poses dataset
    ValueError
        If connections configuration is invalid
    KeyError
        If template name is not found in TEMPLATES

    Notes
    -----
    The skeleton is rendered as a Vectors layer which displays lines
    connecting the specified keypoints. The layer automatically
    integrates with napari's time slider for playback.
    """
    # Validate dataset
    _validate_dataset(dataset)

    # Resolve skeleton configuration
    skeleton_state = _resolve_skeleton_state(connections, dataset)

    # Validate the state
    is_valid, errors = skeleton_state.validate()
    if not is_valid:
        error_msg = "\n".join(errors)
        raise ValueError(f"Invalid skeleton state:\n{error_msg}")

    # Create and prepare renderer
    renderer = PrecomputedRenderer(
        dataset=dataset,
        connections=skeleton_state.connections,
        edge_colors=skeleton_state.edge_colors,
        edge_widths=skeleton_state.edge_widths,
    )
    renderer.prepare()

    # Get vectors and validate
    vectors = renderer.vectors
    if vectors is None or len(vectors) == 0:
        raise ValueError(
            "No valid skeleton vectors could be computed. "
            "Check that keypoints have valid (non-NaN) positions."
        )

    # Build color array using edge indices from renderer
    if renderer.edge_indices is None:
        raise ValueError("Renderer edge_indices is None after preparation")
    vector_colors = _build_vector_colors(
        skeleton_state, renderer.edge_indices
    )

    # Set up layer properties
    vector_kwargs = {
        "edge_color": vector_colors,
        "edge_width": 2.0,
        "name": name,
        "opacity": 0.8,
    }
    vector_kwargs.update(kwargs)

    # Add vectors layer to viewer
    return viewer.add_vectors(vectors, **vector_kwargs)


def load_skeleton_config(path: str) -> dict[str, Any]:
    """Load skeleton configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file

    Returns
    -------
    dict
        Skeleton configuration dictionary
    """
    return load_yaml_config(path)


def save_skeleton_config(config: dict[str, Any], path: str) -> None:
    """Save skeleton configuration to a YAML file.

    Parameters
    ----------
    config : dict
        Skeleton configuration dictionary
    path : str
        Path where the YAML file will be saved
    """
    save_yaml_config(config, path)
