"""Shared binning for two-dimensional occupancy histograms."""

import numpy as np


def occupancy_bin_edges(
    x: np.ndarray, y: np.ndarray, bins: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin edges spanning all finite pairs of positions.

    Empty inputs use a unit range so callers can represent missing tracks
    without choosing a different grid for each track.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return (
            np.linspace(0, 1, bins[0] + 1),
            np.linspace(0, 1, bins[1] + 1),
        )
    return (
        np.histogram_bin_edges(x[valid], bins=bins[0]),
        np.histogram_bin_edges(y[valid], bins=bins[1]),
    )
