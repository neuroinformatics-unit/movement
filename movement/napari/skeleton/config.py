"""Configuration schema and I/O for skeleton visualization."""

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, ...]:
    """Convert hex color to RGBA tuple.

    Parameters
    ----------
    hex_color : str
        Hex color string (e.g., "#FF0000" or "FF0000")
    alpha : float, optional
        Alpha channel value (0-1), by default 1.0

    Returns
    -------
    tuple of float
        RGBA values normalized to 0-1 range

    """
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


def rgba_to_hex(rgba: tuple[float, ...]) -> str:
    """Convert RGBA tuple to hex color string.

    Parameters
    ----------
    rgba : tuple of float
        RGBA values in 0-1 range

    Returns
    -------
    str
        Hex color string (e.g., "#FF0000")

    """
    r, g, b = (int(c * 255) for c in rgba[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def connections_to_edge_indices(
    connections: list[dict[str, Any]], keypoint_names: list[str]
) -> list[tuple[int, int]]:
    """Convert connection definitions to keypoint index pairs.

    Parameters
    ----------
    connections : list of dict
        List of connection dictionaries with 'start' and 'end' keys
    keypoint_names : list of str
        List of keypoint names in the dataset

    Returns
    -------
    list of tuple of int
        List of (start_idx, end_idx) pairs

    Raises
    ------
    ValueError
        If a keypoint name in connections is not found in keypoint_names

    """
    edges = []
    for conn in connections:
        start_name = conn["start"]
        end_name = conn["end"]

        if start_name not in keypoint_names:
            raise ValueError(
                f"Start keypoint '{start_name}' not found "
                f"in dataset keypoints: {keypoint_names}"
            )
        if end_name not in keypoint_names:
            raise ValueError(
                f"End keypoint '{end_name}' not found "
                f"in dataset keypoints: {keypoint_names}"
            )

        start_idx = keypoint_names.index(start_name)
        end_idx = keypoint_names.index(end_name)
        edges.append((start_idx, end_idx))

    return edges


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load skeleton configuration from YAML file.

    Parameters
    ----------
    path : str or Path
        Path to YAML configuration file

    Returns
    -------
    dict
        Skeleton configuration dictionary

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def save_yaml_config(config: dict[str, Any], path: str | Path) -> None:
    """Save skeleton configuration to YAML file.

    Parameters
    ----------
    config : dict
        Skeleton configuration dictionary
    path : str or Path
        Path where YAML file will be saved

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(
    config: dict[str, Any], dataset: xr.Dataset
) -> tuple[bool, list[str]]:
    """Validate skeleton configuration against a dataset.

    Parameters
    ----------
    config : dict
        Skeleton configuration to validate
    dataset : xr.Dataset
        Movement dataset to validate against

    Returns
    -------
    tuple of (bool, list of str)
        (is_valid, error_messages)
        If valid, error_messages will be empty

    """
    errors = []

    # Check required keys
    if "connections" not in config:
        errors.append("Config missing required key: 'connections'")
        return False, errors

    # Get keypoint names from dataset
    if "keypoints" not in dataset.coords:
        errors.append("Dataset has no 'keypoints' coordinate")
        return False, errors

    keypoint_names = list(dataset.coords["keypoints"].values)

    # Validate each connection
    for i, conn in enumerate(config["connections"]):
        if "start" not in conn:
            errors.append(f"Connection {i} missing 'start' keypoint")
        elif conn["start"] not in keypoint_names:
            errors.append(
                f"Connection {i}: keypoint '{conn['start']}' not found "
                f"in dataset"
            )

        if "end" not in conn:
            errors.append(f"Connection {i} missing 'end' keypoint")
        elif conn["end"] not in keypoint_names:
            errors.append(
                f"Connection {i}: keypoint '{conn['end']}' not found "
                f"in dataset"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def config_to_arrays(
    config: dict[str, Any], keypoint_names: list[str]
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, list[str]]:
    """Convert configuration dict to arrays for renderer.

    Parameters
    ----------
    config : dict
        Skeleton configuration dictionary
    keypoint_names : list of str
        List of keypoint names from the dataset

    Returns
    -------
    tuple
        (connections, edge_colors, edge_widths, segment_labels)
        - connections: list of (start_idx, end_idx) tuples
        - edge_colors: array of RGBA colors, shape (n_edges, 4)
        - edge_widths: array of widths, shape (n_edges,)
        - segment_labels: list of segment names

    """
    connections = connections_to_edge_indices(
        config["connections"], keypoint_names
    )

    n_edges = len(connections)
    edge_colors = np.zeros((n_edges, 4))
    edge_widths = np.zeros(n_edges)
    segment_labels = []

    for i, conn in enumerate(config["connections"]):
        # Parse color (defaults to white if not specified)
        color_str = conn.get("color", "#FFFFFF")
        edge_colors[i] = hex_to_rgba(color_str)

        # Parse width (defaults to 2.0 if not specified)
        edge_widths[i] = conn.get("width", 2.0)

        # Parse segment label (defaults to empty string)
        segment_labels.append(conn.get("segment", ""))

    return connections, edge_colors, edge_widths, segment_labels
