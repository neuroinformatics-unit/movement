"""Load data from various frameworks into ``movement``."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import xarray as xr

from movement.io.load_bboxes import from_via_tracks_file
from movement.io.load_poses import (
    from_anipose_file,
    from_dlc_file,
    from_lp_file,
    from_nwb_file,
    from_sleap_file,
)
from movement.utils.logging import logger

_REGISTRY: dict[str, Callable[..., xr.Dataset]] = {
    "DeepLabCut": from_dlc_file,
    "SLEAP": from_sleap_file,
    "LightningPose": from_lp_file,
    "Anipose": from_anipose_file,
    "NWB": from_nwb_file,
    "VIA-tracks": from_via_tracks_file,
}


def from_file(
    file_path: Path | str,
    source_software: Literal[
        "DeepLabCut",
        "SLEAP",
        "LightningPose",
        "Anipose",
        "NWB",
        "VIA-tracks",
    ],
    fps: float | None = None,
    **kwargs,
) -> xr.Dataset:
    """Create a ``movement`` dataset from any supported file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing predicted poses or tracked bounding boxes.
        The file format must be among those supported by the
        :mod:`movement.io.load_poses` or
        :mod:`movement.io.load_bboxes` modules. Based on
        the value of ``source_software``, the appropriate loading function
        will be called.
    source_software : {"DeepLabCut", "SLEAP", "LightningPose", "Anipose", \
        "NWB", "VIA-tracks"}
        The source software of the file.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
        This argument is ignored when ``source_software`` is "NWB", as the
        frame rate will be directly read or estimated from metadata in
        the NWB file.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the software-specific
        loading functions that are listed under "See Also".

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.


    See Also
    --------
    movement.io.load_poses.from_dlc_file
    movement.io.load_poses.from_sleap_file
    movement.io.load_poses.from_lp_file
    movement.io.load_poses.from_anipose_file
    movement.io.load_poses.from_nwb_file
    movement.io.load_bboxes.from_via_tracks_file

    Examples
    --------
    >>> from movement.io import load
    >>> ds = load.from_file(
    ...     "path/to/file.h5", source_software="DeepLabCut", fps=30
    ... )

    """
    if source_software not in _REGISTRY:
        raise logger.error(
            ValueError(f"Unsupported source software: {source_software}")
        )
    if source_software == "NWB":
        if fps is not None:
            logger.warning(
                "The fps argument is ignored when loading from an NWB file. "
                "The frame rate will be directly read or estimated from "
                "metadata in the file."
            )
        return _REGISTRY[source_software](file_path, **kwargs)
    return _REGISTRY[source_software](file_path, fps, **kwargs)


def from_multiview_files(
    file_path_dict: dict[str, Path | str],
    source_software: Literal[
        "DeepLabCut", "SLEAP", "LightningPose", "Anipose", "NWB", "VIA-tracks"
    ],
    fps: float | None = None,
    **kwargs,
) -> xr.Dataset:
    """Load and merge data from multiple files representing different views.

    Parameters
    ----------
    file_path_dict : dict[str, Union[Path, str]]
        A dict whose keys are the view names and values are the paths to load.
    source_software : {"DeepLabCut", "SLEAP", "LightningPose", "Anipose", \
        "NWB", "VIA-tracks"}
        The source software of the file.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
        This argument is ignored when ``source_software`` is "NWB", as the
        frame rate will be directly read or estimated from metadata in
        the NWB file.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the software-specific
        loading functions that are listed under "See Also".

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing data concatenated along a new
        ``view`` dimension.

    Notes
    -----
    The attributes of the resulting dataset will be taken from the first
    dataset specified in ``file_path_dict``. This is the default
    behaviour of :func:`xarray.concat` used under the hood.

    """
    views_list = list(file_path_dict.keys())
    new_coord_views = xr.DataArray(views_list, dims="view")
    dataset_list = [
        from_file(f, source_software=source_software, fps=fps, **kwargs)
        for f in file_path_dict.values()
    ]
    return xr.concat(dataset_list, dim=new_coord_views)
