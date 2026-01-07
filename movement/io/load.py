"""Load data from various frameworks into ``movement``."""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Literal, Protocol, cast

import pynwb
import xarray as xr

from movement.utils.logging import logger
from movement.validators.files import ValidFile


class LoaderProtocol(Protocol):
    """Protocol for loader functions."""

    def __call__(
        self, file: Path | str | pynwb.file.NWBFile, *args, **kwargs
    ) -> xr.Dataset:
        """Load data from a file.

        Parameters
        ----------
        file
            Path to the file or a :class:`pynwb.file.NWBFile` object.
        *args
            Additional positional arguments for the loader.
        **kwargs
            Additional keyword arguments for the loader.

        Returns
        -------
        xarray.Dataset
            The loaded dataset.

        """
        ...


_REGISTRY: dict[str, LoaderProtocol] = {}


def register_loader(
    source_software: str,
    *,
    expected_suffix: list[str],
    expected_permission: Literal["r", "w", "rw"] = "r",
) -> Callable[[LoaderProtocol], LoaderProtocol]:
    """Register a loader function for a given source software.

    Parameters
    ----------
    source_software
        The name of the source software.
    expected_suffix
        Expected suffix(es) for the file. If an empty list (default), this
        check is skipped.
    expected_permission
        Expected access permission(s) for the file. If "r", the file is
        expected to be readable. If "w", the file is expected to be writable.
        If "rw", the file is expected to be both readable and writable.
        Default: "r".

    Returns
    -------
    Callable
        A decorator that registers the loader function.

    Examples
    --------
    >>> from movement.io.load import register_loader
    >>> @register_loader("DeepLabCut", expected_suffix=[".h5", ".csv"])
    ... def from_dlc_file(file_path: str, fps=None, **kwargs):
    ...     pass

    """

    def decorator(loader_fn: LoaderProtocol) -> LoaderProtocol:
        @wraps(loader_fn)
        def wrapper(
            file_path: Path | str | pynwb.file.NWBFile, *args, **kwargs
        ) -> xr.Dataset:
            if isinstance(file_path, (str, Path)):
                file_path = ValidFile(
                    file_path,
                    expected_suffix=expected_suffix,
                    expected_permission=expected_permission,
                ).path
            return loader_fn(file_path, *args, **kwargs)

        _REGISTRY[source_software] = cast(LoaderProtocol, wrapper)
        return cast(LoaderProtocol, wrapper)

    return decorator


def from_file(
    file: Path | str | pynwb.file.NWBFile,
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
    file
        Path to the file containing predicted poses or tracked bounding boxes.
        If ``source software`` is "NWB", this can also be a
        :class:`pynwb.file.NWBFile` object.
        The file format must be among those supported by the
        :mod:`movement.io.load_poses` or
        :mod:`movement.io.load_bboxes` modules. Based on
        the value of ``source_software``, the appropriate loading function
        will be called.
    source_software
        The source software of the file.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
        This argument is ignored when ``source_software`` is "NWB", as the
        frame rate will be directly read or estimated from metadata in
        the NWB file.
    **kwargs
        Additional keyword arguments to pass to the software-specific
        loading functions that are listed under "See Also".

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose or bounding box tracks,
        confidence scores, and associated metadata.


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
        return _REGISTRY[source_software](file, **kwargs)
    return _REGISTRY[source_software](file, fps, **kwargs)


def from_multiview_files(
    file_dict: dict[str, Path | str],
    source_software: Literal[
        "DeepLabCut", "SLEAP", "LightningPose", "Anipose", "NWB", "VIA-tracks"
    ],
    fps: float | None = None,
    **kwargs,
) -> xr.Dataset:
    """Load and merge data from multiple files representing different views.

    Parameters
    ----------
    file_dict
        A dict whose keys are the view names and values are the paths to load.
    source_software
        The source software of the file.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
        This argument is ignored when ``source_software`` is "NWB", as the
        frame rate will be directly read or estimated from metadata in
        the NWB file.
    **kwargs
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
    views_list = list(file_dict.keys())
    new_coord_views = xr.DataArray(views_list, dims="view")
    dataset_list = [
        from_file(f, source_software=source_software, fps=fps, **kwargs)
        for f in file_dict.values()
    ]
    return xr.concat(dataset_list, dim=new_coord_views)
