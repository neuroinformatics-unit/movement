"""Save ``movement`` datasets to various file formats."""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Concatenate, Literal, ParamSpec, Protocol, cast

import xarray as xr

from movement.utils.logging import logger
from movement.validators.datasets import DS_TYPE_VALIDATORS, DsType
from movement.validators.files import validate_file_path

type TargetSoftware = Literal[
    "netCDF",
    "DeepLabCut",
    "SLEAP",
    "LightningPose",
    "NWB",
    "VIA-tracks",
]

P = ParamSpec("P")


class WriterProtocol(Protocol):
    """Protocol for writer functions to be registered via ``register_writer``.

    All writer functions registered via :func:`register_writer` must conform
    to this protocol. Writers must accept a ``movement`` dataset and a file
    path, and save the dataset directly to disk. Additional positional and
    keyword arguments are allowed.

    See Also
    --------
    register_writer : Decorator for registering writer functions.

    """

    def __call__(
        self, ds: xr.Dataset, file: str | Path, *args, **kwargs
    ) -> None:
        """Save a ``movement`` dataset to a file.

        Parameters
        ----------
        ds
            The ``movement`` dataset to save.
        file
            Path to the file to save the dataset to.
        *args
            Additional positional arguments for the writer.
        **kwargs
            Additional keyword arguments for the writer.

        """
        ...


_WRITER_REGISTRY: dict[str, WriterProtocol] = {}
_WRITER_DS_TYPE_REGISTRY: dict[str, DsType | None] = {}


def register_writer(
    target_software: str,
    *,
    ds_type: DsType | None = None,
    suffixes: set[str] | None = None,
) -> Callable[
    [Callable[Concatenate[xr.Dataset, str | Path, P], None]],
    Callable[Concatenate[xr.Dataset, str | Path, P], None],
]:
    """Register a writer function for a given target software.

    The decorator also handles dataset and file path validation: the
    dataset is checked against its expected type (``ds_type``, or if
    unset, the type inferred from ``ds.attrs["ds_type"]``), and the file
    path is checked for write permission and optional suffixes, before
    the writer function is called.

    Parameters
    ----------
    target_software
        The name of the target software to save to.
    ds_type
        The ``movement`` dataset type the writer is restricted to. If None
        (default), the writer (e.g., ``movement``'s native netCDF format)
        accepts either type, and the expected type is instead inferred from the
        dataset's ``ds_type`` attribute at save time.
    suffixes
        The set of valid file suffixes (e.g. ``{".h5", ".csv"}``) for this
        writer. If None (default), the file path is still validated for write
        permission, but its suffix is not checked.

    Returns
    -------
    collections.abc.Callable
        A decorator that registers the writer function.

    Notes
    -----
    The ``file`` argument passed to the decorated writer function will be
    a validated :class:`pathlib.Path` object, instead of the original
    file path.

    Examples
    --------
    >>> from movement.io.save import register_writer
    >>> @register_writer("SLEAP", ds_type="poses", suffixes={".h5"})
    ... def to_sleap_analysis_file(ds: xr.Dataset, file: str | Path):
    ...     pass

    """

    def decorator(
        writer_fn: Callable[Concatenate[xr.Dataset, str | Path, P], None],
    ) -> Callable[Concatenate[xr.Dataset, str | Path, P], None]:
        @wraps(writer_fn)
        def wrapper(
            ds: xr.Dataset, file: str | Path, *args: P.args, **kwargs: P.kwargs
        ) -> None:
            _validate_ds_type(ds, target_software)
            valid_file = validate_file_path(
                file, permission="w", suffixes=suffixes
            )
            return writer_fn(ds, valid_file, *args, **kwargs)

        _WRITER_REGISTRY[target_software] = cast("WriterProtocol", wrapper)
        _WRITER_DS_TYPE_REGISTRY[target_software] = ds_type
        return wrapper

    return decorator


def save_dataset(
    ds: xr.Dataset,
    file: str | Path,
    target_software: TargetSoftware | None = None,
    **kwargs,
) -> None:
    """Save a ``movement`` dataset to a file in any supported format.

    Parameters
    ----------
    ds
        The ``movement`` dataset to save.
    file
        Path to the file to save the dataset to. The required file extension
        depends on the target format.
    target_software
        The format to save the dataset in. If None (default), the dataset is
        saved in ``movement``'s native netCDF format (``.nc``). Otherwise, one
        of the :ref:`supported third-party formats <target-supported-formats>`.
    **kwargs
        Additional keyword arguments passed to the format-specific writer
        (e.g. ``split_individuals`` for DeepLabCut, ``config`` for NWB, or any
        :meth:`xarray.Dataset.to_netcdf` argument for netCDF).

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``ds`` is not an :class:`xarray.Dataset`.
    ValueError
        If ``target_software`` is not a supported format, or if it is
        incompatible with the dataset's ``ds_type`` (e.g. saving a bounding
        boxes dataset to a poses-only format).

    See Also
    --------
    movement.io.save_poses
    movement.io.save_bboxes

    Notes
    -----
    For NWB, each individual is written to a separate file (as required by the
    NWB format). For a single-individual dataset, the dataset is written to
    ``file``. For a multi-individual dataset, each NWBFile's ``identifier``
    (by default the individual's name, but configurable via
    :class:`~movement.io.nwb.NWBFileSaveConfig`) is appended to the file
    path, just before the extension, e.g. ``"/path/to/file_id_0.nwb"``.

    Examples
    --------
    Save a dataset to ``movement``'s native netCDF format:

    >>> from movement.io import save_dataset
    >>> save_dataset(ds, "/path/to/file.nc")

    Save a poses dataset to a DeepLabCut .h5 file:

    >>> save_dataset(ds, "/path/to/file.h5", target_software="DeepLabCut")

    Save a bounding boxes dataset to a VIA-tracks .csv file:

    >>> save_dataset(ds, "/path/to/file.csv", target_software="VIA-tracks")

    Save a poses dataset to NWB file(s), with custom metadata:

    >>> from movement.io.nwb import NWBFileSaveConfig
    >>> config = NWBFileSaveConfig(
    ...     nwbfile_kwargs={"session_description": "foraging session"}
    ... )
    >>> save_dataset(
    ...     ds, "/path/to/file.nwb", target_software="NWB", config=config
    ... )

    """
    target = target_software if target_software is not None else "netCDF"
    if target not in _WRITER_REGISTRY:
        supported = ", ".join(_WRITER_REGISTRY)
        raise logger.error(
            ValueError(
                f"Unsupported target_software for saving: '{target}'. "
                f"Supported values are: {supported}."
            )
        )
    _WRITER_REGISTRY[target](ds, file, **kwargs)


def _validate_ds_type(ds: xr.Dataset, target: str) -> None:
    """Check that ``ds`` is a valid dataset for the given save ``target``.

    For targets restricted to a specific ``movement`` dataset type (poses or
    bounding boxes), the corresponding validator's ``validate()`` classmethod
    is used to check that ``ds`` has the required variables and dimensions
    for that type.

    For targets compatible with either type (currently only netCDF), the
    expected type is inferred from ``ds.attrs["ds_type"]``.
    """
    expected = _WRITER_DS_TYPE_REGISTRY[target]
    if expected is None:  # netCDF: determine the expected type from ds_type
        try:
            expected = ds.attrs.get("ds_type")
        except AttributeError:
            # let validator raise TypeError for non-xr.Dataset
            expected = "poses"
        if expected not in DS_TYPE_VALIDATORS:
            raise logger.error(
                ValueError(
                    f"Cannot save to '{target}': expected `ds` to have a "
                    f"'ds_type' attribute of 'poses' or 'bboxes', but got "
                    f"{expected!r}."
                )
            )
    DS_TYPE_VALIDATORS[expected].validate(ds)


@register_writer("netCDF", suffixes={".nc"})
def _to_netcdf_file(ds: xr.Dataset, file: str | Path, **kwargs) -> None:
    """Save a ``movement`` dataset to a netCDF (.nc) file."""
    ds.to_netcdf(file, **kwargs)
    logger.info(f"Saved dataset to {file}.")
