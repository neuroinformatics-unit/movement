"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from movement.io import load_poses
from movement.napari.convert import ds_to_napari_tracks

# get logger
logger = logging.getLogger(__name__)


def napari_get_reader(
    path: Union[str, list[str]],
) -> Optional[Callable]:
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if a list, check only the first path (assuming they are all the same)
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    accepted_extensions = [".h5", ".hdf5", ".slp", ".csv"]
    if Path(path).suffix not in accepted_extensions:
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(
    path: Union[str, list[str]],
) -> list[tuple[np.ndarray[Any, Any], dict[str, Any], str]]:
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a str or a list of strs
    file_paths = [path] if isinstance(path, str) else path

    layers = []

    for file_path in file_paths:
        file_name = Path(file_path).name
        logger.debug(f"Trying to load pose tracks from {file_name}.")
        if file_name.startswith("DLC"):
            ds = load_poses.from_dlc_file(file_path, fps=30)
        elif file_name.startswith("SLEAP"):
            ds = load_poses.from_sleap_file(file_path, fps=30)
        else:  # starts with 'LP'
            ds = load_poses.from_lp_file(file_path, fps=30)

        data, props = ds_to_napari_tracks(ds)
        logger.info("Converted pose tracks to a napari Tracks array.")
        logger.debug(f"Tracks data shape: {data.shape}")
        logger.debug(f"{data[:5, :]}")

        n_individuals = len(np.unique(props["individual"]))
        color_by = "individual" if n_individuals > 1 else "keypoint"

        # kwargs for the napari Tracks layer
        tracks_kwargs = {
            "name": file_name,
            "properties": props,
            "visible": True,
            "tail_width": 5,
            "tail_length": 60,
            "head_length": 0,
            "colormap": "turbo",
            "color_by": color_by,
            "blending": "translucent",
        }

        # Add the napari Tracks layer
        layers.append((data, tracks_kwargs, "tracks"))

        # kwargs for the napari Points layer
        points_kwargs = {
            "name": file_name,
            "properties": props,
            "visible": True,
            "symbol": "ring",
            "size": 15,
            "face_color": "red",
            "edge_width": 0,
            "blending": "translucent",
        }

        # Add the napari Points layer
        layers.append((data[:, 1:], points_kwargs, "points"))

    return layers
