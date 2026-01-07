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

    Examples
    --------
    >>> import napari
    >>> from movement.sample_data import fetch_dataset
    >>> from movement.napari.skeleton import add_skeleton_layer, TEMPLATES
    >>> ds = fetch_dataset("DLC_single-wasp.predictions.h5")
    >>> viewer = napari.Viewer()
    >>> skeleton_layer = add_skeleton_layer(
    ...     viewer,
    ...     ds,
    ...     connections="mouse",  # Use mouse template
    ...     name="mouse_skeleton"
    ... )

    Notes
    -----
    The skeleton is rendered as a Vectors layer which displays lines
    (not solid shapes) connecting the specified keypoints. The layer
    automatically integrates with napari's time slider for playback.
    """
    # Validate dataset
    if dataset.ds_type != "poses":
        raise ValueError(
            "Dataset must be a poses dataset with 'position' and "
            "'keypoints' data"
        )

    if "keypoints" not in dataset.coords:
        raise ValueError("Dataset must have 'keypoints' coordinate")

    # Get keypoint names
    keypoint_names = list(dataset.coords["keypoints"].values)

    # Resolve connections configuration
    if connections is None:
        # Try to load from dataset
        skeleton_state = SkeletonState.from_dataset(dataset)
        if skeleton_state is None:
            raise ValueError(
                "No skeleton configuration found. "
                "Provide 'connections' parameter or use a dataset "
                "with embedded skeleton configuration."
            )
    elif isinstance(connections, str):
        # Load template by name
        if connections not in TEMPLATES:
            available = ", ".join(TEMPLATES.keys())
            raise KeyError(
                f"Template '{connections}' not found. "
                f"Available templates: {available}"
            )
        # Cast template to dict for type checking
        config: dict[str, Any] = dict(TEMPLATES[connections])
        skeleton_state = SkeletonState.from_config(config, dataset)
    elif isinstance(connections, dict):
        # Use provided configuration
        is_valid, errors = validate_config(connections, dataset)
        if not is_valid:
            error_msg = "\n".join(errors)
            raise ValueError(f"Invalid skeleton configuration:\n{error_msg}")
        skeleton_state = SkeletonState.from_config(connections, dataset)
    else:
        raise ValueError(
            "connections must be a dict, str (template name), or None"
        )

    # Validate the state
    is_valid, errors = skeleton_state.validate()
    if not is_valid:
        error_msg = "\n".join(errors)
        raise ValueError(f"Invalid skeleton state:\n{error_msg}")

    # Create renderer
    renderer = PrecomputedRenderer(
        dataset=dataset,
        connections=skeleton_state.connections,
        edge_colors=skeleton_state.edge_colors,
        edge_widths=skeleton_state.edge_widths,
    )

    # Prepare renderer (pre-compute vectors)
    renderer.prepare()

    # Get the skeleton vectors
    vectors = renderer.vectors

    if vectors is None or len(vectors) == 0:
        raise ValueError(
            "No valid skeleton vectors could be computed. "
            "Check that keypoints have valid (non-NaN) positions."
        )

    # Create vector properties for coloring
    # Replicate colors for each vector (napari requires per-vector colors)
    n_edges = len(skeleton_state.connections)
    n_individuals = dataset.sizes["individuals"]
    n_frames = dataset.sizes["time"]

    # Build color array matching vectors
    # Each connection gets its color repeated for all frames/individuals
    vector_colors_list: list[tuple[float, ...]] = []
    for _ in range(n_frames):
        for _ in range(n_individuals):
            for edge_idx in range(n_edges):
                # Get color for this edge
                color = tuple(skeleton_state.edge_colors[edge_idx])
                vector_colors_list.append(color)

    # Filter to only valid vectors (those that were actually created)
    # The renderer skips NaN keypoints, so we need matching length
    if len(vector_colors_list) > len(vectors):
        vector_colors_list = vector_colors_list[: len(vectors)]

    vector_colors = np.array(vector_colors_list)

    # Set default vector layer properties
    vector_kwargs = {
        "edge_color": vector_colors,
        "edge_width": 2.0,  # Default width, can be overridden
        "name": name,
        "opacity": 0.8,
    }

    # Override with user-provided kwargs
    vector_kwargs.update(kwargs)

    # Add vectors layer to viewer
    layer = viewer.add_vectors(vectors, **vector_kwargs)

    return layer


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

    See Also
    --------
    save_skeleton_config : Save configuration to YAML file
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

    See Also
    --------
    load_skeleton_config : Load configuration from YAML file
    """
    save_yaml_config(config, path)
