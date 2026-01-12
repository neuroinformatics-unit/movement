"""Load data from various frameworks into ``movement``."""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Concatenate, Literal, ParamSpec, Protocol, TypeVar, cast

import attrs
import pynwb
import xarray as xr

from movement.utils.logging import logger
from movement.validators.files import ValidFile

TInputFile = TypeVar("TInputFile", Path, str, pynwb.file.NWBFile)
P = ParamSpec("P")


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


_LOADER_REGISTRY: dict[str, LoaderProtocol] = {}


def _get_validator_kwargs(
    validator_cls: type[ValidFile],
    *,
    loader_kwargs: dict,
) -> dict:
    """Extract the relevant kwargs for a given validator class."""
    # Only extract fields that are used in the validator's __init__
    validator_fields = {
        field.name for field in attrs.fields(validator_cls) if field.init
    }
    return {
        field_name: loader_kwargs[field_name]
        for field_name in validator_fields
        if field_name in loader_kwargs
    }


def register_loader(
    source_software: str,
    *,
    file_validators: type[ValidFile] | list[type[ValidFile]] | None = None,
) -> Callable[
    [Callable[Concatenate[ValidFile, P], xr.Dataset]],
    Callable[Concatenate[TInputFile, P], xr.Dataset],
]:
    """Register a loader function for a given source software.

    Parameters
    ----------
    source_software
        The name of the source software.
    file_validators
        File validator(s) to validate the input file path and content.

    Returns
    -------
    Callable
        A decorator that registers the loader function.

    Examples
    --------
    >>> from movement.io.load import register_loader
    >>> from movement.validators.files import (
    ...     ValidDeepLabCutH5,
    ...     ValidDeepLabCutCSV,
    ... )
    >>> @register_loader(
    ...     "DeepLabCut",
    ...     file_validators=[ValidDeepLabCutH5, ValidDeepLabCutCSV],
    ... )
    ... def from_dlc_file(file: str, fps=None, **kwargs):
    ...     pass

    """
    validators_list: list[type[ValidFile]] = (
        [file_validators]
        if file_validators is not None
        and not isinstance(file_validators, list)
        else file_validators or []
    )
    # Map suffixes to file validator classes
    suffix_map: dict[str, type[ValidFile]] = {}
    for validator_cls in validators_list:
        for suffix in getattr(validator_cls, "suffixes", set()):
            suffix_map[suffix] = validator_cls

    def decorator(
        loader_fn: Callable[Concatenate[ValidFile, P], xr.Dataset],
    ) -> Callable[Concatenate[TInputFile, P], xr.Dataset]:
        @wraps(loader_fn)
        def wrapper(file: TInputFile, *args, **kwargs) -> xr.Dataset:
            if isinstance(file, pynwb.file.NWBFile):
                file_suffix = ".nwb"
            else:
                file_suffix = Path(file).suffix
            validator_cls = suffix_map.get(file_suffix)
            if validator_cls is None:
                raise logger.error(
                    ValueError(
                        f"Unsupported format for '{source_software}': ",
                        f"{file_suffix}.",
                    )
                )
            validator_kwargs = _get_validator_kwargs(
                validator_cls, loader_kwargs=kwargs
            )
            # Validate the file
            valid_file = validator_cls(
                file=file,
                **validator_kwargs,  # type: ignore[call-arg]
            )
            return loader_fn(valid_file, *args, **kwargs)

        # Register the loader in the global registry
        _LOADER_REGISTRY[source_software] = cast("LoaderProtocol", wrapper)
        return wrapper

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
    if source_software not in _LOADER_REGISTRY:
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
        return _LOADER_REGISTRY[source_software](file, **kwargs)
    return _LOADER_REGISTRY[source_software](file, fps, **kwargs)


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
