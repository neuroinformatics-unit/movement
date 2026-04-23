"""Load data from various frameworks into ``movement``."""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import (
    Concatenate,
    Final,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
)

import attrs
import pynwb
import xarray as xr

from movement.utils.logging import logger
from movement.validators.files import ValidFile

TInputFile = TypeVar("TInputFile", Path, str, pynwb.file.NWBFile)
P = ParamSpec("P")
type SourceSoftware = Literal[
    "DeepLabCut",
    "SLEAP",
    "LightningPose",
    "Anipose",
    "NWB",
    "VIA-tracks",
]
AMBIGUOUS_DLC_LP_SOURCE_SOFTWARE: Final = "DeepLabCut/LightningPose"

type InferredSourceSoftware = (
    SourceSoftware | Literal["DeepLabCut/LightningPose"]
)
type SourceSoftwareOrAuto = (
    SourceSoftware | Literal["auto", "DeepLabCut/LightningPose"]
)


class LoaderProtocol(Protocol):
    """Protocol for loader functions to be registered via ``register_loader``.

    All loader functions registered via :func:`register_loader` must conform
    to this protocol. Loaders must accept a file path or a
    :class:`pynwb.file.NWBFile` object as their first argument and return an
    :class:`xarray.Dataset` containing pose tracks or bounding box tracks.
    Additional positional and keyword arguments are allowed.

    See Also
    --------
    register_loader : Decorator for registering loader functions.

    """

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
_LOADER_VALIDATORS_REGISTRY: dict[SourceSoftware, list[type[ValidFile]]] = {}


def infer_source_software(
    file: Path | str | pynwb.file.NWBFile,
    **loader_kwargs,
) -> InferredSourceSoftware:
    """Infer the ``source_software`` from a given input file.

    Tries the file validators registered for each built-in loader and
    returns the name of the matching source software. If multiple candidates
    match (i.e. the file format is shared across software packages, as with
    DeepLabCut-style CSV files), a combined label is returned (e.g.
    ``"DeepLabCut/LightningPose"``).

    Parameters
    ----------
    file
        Input file path or an :class:`pynwb.file.NWBFile` object.
    **loader_kwargs
        Optional keyword arguments forwarded to file validators (when they
        accept additional parameters, e.g. VIA-tracks `frame_regexp`).

    Returns
    -------
    InferredSourceSoftware
        The inferred source software name.

    Raises
    ------
    ValueError
        If no registered validator matches the file or if multiple
        ``source_software`` candidates (beyond the expected
        DeepLabCut/LightningPose case) match the file.

    """
    # If it's an NWBFile object, we can immediately return "NWB"
    if isinstance(file, pynwb.file.NWBFile):
        return "NWB"
    file_path = Path(file)

    # Try all registered validators and keep track of matching candidates
    candidates: list[SourceSoftware] = []

    for source_sw, validators_list in _LOADER_VALIDATORS_REGISTRY.items():
        suffix_map = _build_suffix_map(validators_list)
        try:
            # Temporarily disable the logger to avoid flooding the console
            # with expected validation errors during candidate testing
            logger.disable("movement")
            _validate_file(
                file_path,
                suffix_map,
                source_sw,
                loader_kwargs=loader_kwargs,
            )
        except (OSError, TypeError, ValueError):
            continue
        finally:
            logger.enable("movement")  # re-store logging
        candidates.append(source_sw)

    # If exactly one candidate matches, return it.
    if len(candidates) == 1:
        return candidates[0]

    # DeepLabCut and LightningPose currently share the same CSV validator.
    # If both match, we return a combined label to indicate the ambiguity.
    if set(candidates) == {"DeepLabCut", "LightningPose"}:
        return AMBIGUOUS_DLC_LP_SOURCE_SOFTWARE

    # In every other case (no candidates or multiple unexpected candidates),
    # we cannot infer a valid source software.
    raise logger.error(
        ValueError(
            f"Could not infer source_software from file '{file_path}'. "
            "Verify that the file format is supported by movement or "
            "try passing an explicit source_software value to load_dataset."
        )
    )


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


def _build_suffix_map(
    validators_list: list[type[ValidFile]],
) -> dict[str, type[ValidFile]]:
    """Build a mapping of file suffixes to validator classes."""
    suffix_map: dict[str, type[ValidFile]] = {}
    for validator_cls in validators_list:
        for suffix in getattr(validator_cls, "suffixes", set()):
            suffix_map[suffix] = validator_cls
    return suffix_map


def _validate_file[T: (Path, str, pynwb.file.NWBFile)](
    file: T,
    suffix_map: dict[str, type[ValidFile]],
    source_software: SourceSoftware,
    loader_kwargs: dict | None = None,
) -> ValidFile:
    """Validate the input file using the appropriate validator.

    Parameters
    ----------
    file
        The file path or NWBFile object to validate.
    suffix_map
        Mapping of file suffixes to validator classes.
    source_software
        The source software name (for error messages).
    loader_kwargs
        Additional arguments from the loader function to pass to
        the validator.

    Returns
    -------
    ValidFile
        A validated file instance.

    Raises
    ------
    ValueError
        If the file format is not supported.

    """
    if isinstance(file, pynwb.file.NWBFile):
        file_suffix = ".nwb"
    else:
        file_suffix = Path(file).suffix
    validator_cls = suffix_map.get(file_suffix)
    if validator_cls is None:
        raise logger.error(
            ValueError(
                f"Unsupported format for '{source_software}': {file_suffix}."
            )
        )

    validator_kwargs = _get_validator_kwargs(
        validator_cls, loader_kwargs=loader_kwargs or {}
    )
    return validator_cls(
        file=file,
        **validator_kwargs,  # type: ignore[call-arg]
    )


def register_loader(
    source_software: SourceSoftware,
    *,
    file_validators: type[ValidFile] | list[type[ValidFile]] | None = None,
) -> Callable[
    [Callable[Concatenate[TInputFile, P], xr.Dataset]],
    Callable[Concatenate[TInputFile, P], xr.Dataset],
]:
    """Register a loader function for a given source software.

    The decorator also handles file validation using any provided
    file validator(s).

    Parameters
    ----------
    source_software
        The name of the source software.
    file_validators
        File validator(s) to validate the input file path and content.

    Returns
    -------
    collections.abc.Callable
        A decorator that registers the loader function.

    Notes
    -----
    If file validators are provided, the ``file`` argument passed to the
    decorated loader function will be an instance of the appropriate
    :class:`movement.validators.files.ValidFile` subclass, instead of the
    original file path or :class:`pynwb.file.NWBFile` object.

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
    ... def from_dlc_file(file: str | Path, fps=None, **kwargs):
    ...     pass

    """
    validators_list: list[type[ValidFile]] = (
        [file_validators]
        if file_validators is not None
        and not isinstance(file_validators, list)
        else file_validators or []
    )
    _LOADER_VALIDATORS_REGISTRY[source_software] = validators_list
    # Map suffixes to validator classes
    suffix_map = _build_suffix_map(validators_list)

    def decorator(
        loader_fn: Callable[Concatenate[TInputFile, P], xr.Dataset],
    ) -> Callable[Concatenate[TInputFile, P], xr.Dataset]:
        @wraps(loader_fn)
        def wrapper(file: TInputFile, *args, **kwargs) -> xr.Dataset:
            if not validators_list:
                return loader_fn(file, *args, **kwargs)

            valid_file = _validate_file(
                file, suffix_map, source_software, kwargs
            )
            return loader_fn(valid_file, *args, **kwargs)  # type: ignore[arg-type]

        # Register the loader in the global registry
        _LOADER_REGISTRY[source_software] = cast("LoaderProtocol", wrapper)
        return wrapper

    return decorator


def load_dataset(
    file: Path | str | pynwb.file.NWBFile,
    source_software: SourceSoftwareOrAuto = "auto",
    fps: float | None = None,
    **kwargs,
) -> xr.Dataset:
    """Create a ``movement`` dataset from any supported third-party file.

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
        The source software of the file. If set to ``"auto"`` (default),
        it is inferred from the file format via :func:`infer_source_software`.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
        This argument is ignored when ``source_software`` is "NWB", as the
        frame rate will be directly read or estimated from metadata in
        the NWB file.
    **kwargs
        Additional keyword arguments to pass to the software-specific
        loading functions in modules listed under "See Also".

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose or bounding box tracks,
        confidence scores, and associated metadata.


    See Also
    --------
    movement.io.load_poses
    movement.io.load_bboxes

    Examples
    --------
    >>> from movement.io import load_dataset
    >>> ds = load_dataset(
    ...     "path/to/file.h5", source_software="DeepLabCut", fps=30
    ... )

    """
    if source_software == "auto":
        source_software = infer_source_software(file, **kwargs)

    if source_software == AMBIGUOUS_DLC_LP_SOURCE_SOFTWARE:
        ds = _LOADER_REGISTRY["DeepLabCut"](file, fps, **kwargs)
        ds.attrs["source_software"] = AMBIGUOUS_DLC_LP_SOURCE_SOFTWARE
        return ds

    if source_software not in _LOADER_REGISTRY:
        valid_options = ", ".join(sorted(_LOADER_REGISTRY.keys()))
        raise logger.error(
            ValueError(
                f"Unsupported source software: '{source_software}'. "
                f"Supported options are: {valid_options}"
            )
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


def load_multiview_dataset(
    file_dict: dict[str, Path | str],
    source_software: SourceSoftwareOrAuto = "auto",
    fps: float | None = None,
    **kwargs,
) -> xr.Dataset:
    """Load and merge data from multiple files representing different views.

    Parameters
    ----------
    file_dict
        A dict whose keys are the view names and values are the paths to load.
    source_software
        The source software of the files. If set to ``"auto"`` (default),
        it is inferred from the file format via :func:`infer_source_software`.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
        This argument is ignored when ``source_software`` is "NWB", as the
        frame rate will be directly read or estimated from metadata in
        the NWB file.
    **kwargs
        Additional keyword arguments to pass to the software-specific
        loading functions in modules listed under "See Also".

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

    See Also
    --------
    movement.io.load_poses
    movement.io.load_bboxes

    """
    views_list = list(file_dict.keys())
    new_coord_views = xr.DataArray(views_list, dims="view")
    dataset_list = [
        load_dataset(f, source_software=source_software, fps=fps, **kwargs)
        for f in file_dict.values()
    ]
    return xr.concat(dataset_list, dim=new_coord_views)
