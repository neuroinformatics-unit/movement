"""Utility functions to safely configure napari layers."""

import logging

from napari.layers import Tracks
from napari.viewer import Viewer

logger = logging.getLogger(__name__)


def set_tracks_color_by(
    layer: Tracks,
    preferred: str,
    fallback: str = "track_id",
    viewer: Viewer = None,
):
    """Safely sets the 'color_by' property for a Napari Tracks layer.

    Falls back to a default property if the preferred one is not available.

    Parameters
    ----------
    layer : napari.layers.Tracks
        The Tracks layer on which to set color_by.
    preferred : str
        The desired feature to color by.
    fallback : str, optional
        The fallback feature to use if the preferred one is not available.
    viewer : napari.Viewer, optional
        If provided, will display a GUI message in the Napari overlay.

    """
    if preferred in layer.features.columns:
        layer.color_by = preferred
        logger.debug(f"[TracksColor] Successfully colored by '{preferred}'.")
    else:
        layer.color_by = fallback
        logger.debug(
            f"[TracksColor] '{preferred}' not found in features. "
            f"Falling back to '{fallback}'."
        )
        if viewer:
            viewer.text_overlay.text = (
                f"Note: '{preferred}' not found. "
                f"Using '{fallback}' to color tracks."
            )
