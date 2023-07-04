"""
Functions to convert between different formats,
e.g. from DeepLabCut to SLEAP and vice versa.
"""
import logging

import pandas as pd

# get logger
logger = logging.getLogger(__name__)


def sleap_poses_to_dlc_df(pose_tracks: dict) -> pd.DataFrame:
    """Convert pose tracking data from SLEAP labels to a DeepLabCut-style
    DataFrame with multi-index columns. See Notes for details.

    Parameters
    ----------
    pose_tracks : dict
        Dictionary containing `pose_tracks`, `node_names` and `track_names`.
        This dictionary is returned by `io.load_poses.from_sleap`.

    Returns
    -------
    pandas DataFrame
        DataFrame containing pose tracks in DLC style, with the multi-index
        columns ("scorer", "individuals", "bodyparts", "coords").

    Notes
    -----
    Correspondence between SLEAP and DLC terminology:
    - DLC "scorer" has no equivalent in SLEAP, so we assign it to "SLEAP"
    - DLC "individuals" are the names of SLEAP "tracks"
    - DLC "bodyparts" are the names of SLEAP "nodes" (i.e. the keypoints)
    - DLC "coords" are referred to in SLEAP as "dims"
        (i.e. "x" coord + "y" coord + "confidence/likelihood")
    - DLC reports "likelihood" while SLEAP reports "confidence".
        These both measure the point-wise prediction confidence but do not
        have the same range and cannot be compared between the two frameworks.
    """

    # Get the number of frames, tracks, nodes and dimensions
    n_frames, n_tracks, n_nodes, n_dims = pose_tracks["tracks"].shape
    # Use the DLC terminology: scorer, individuals, bodyparts, coords
    # The assigned scorer is always "DeepLabCut"
    scorer = ["SLEAP"]
    individuals = pose_tracks["track_names"]
    bodyparts = pose_tracks["node_names"]
    coords = ["x", "y", "likelihood"]

    # Create the DLC-style multi-index dataframe
    index_levels = ["scorer", "individuals", "bodyparts", "coords"]
    columns = pd.MultiIndex.from_product(
        [scorer, individuals, bodyparts, coords], names=index_levels
    )
    df = pd.DataFrame(
        data=pose_tracks["tracks"].reshape(n_frames, -1),
        index=pd.RangeIndex(0, n_frames),
        columns=columns,
        dtype=float,
    )

    # Log the conversion
    logger.info(
        f"Converted SLEAP pose tracks to DLC-style DataFrame "
        f"with shape {df.shape}"
    )
    return df
