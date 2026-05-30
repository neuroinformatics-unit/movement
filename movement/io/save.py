"""Save ``movement`` datasets to various file formats."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pynwb
import xarray as xr

from movement.io import save_bboxes, save_poses
from movement.utils.logging import logger
from movement.validators.files import validate_file_path

type SaveTarget = Literal[
    "netCDF",
    "DeepLabCut",
    "SLEAP",
    "LightningPose",
    "NWB",
    "VIA-tracks",
]

# Mapping of pose-format names to the writer that handles them. Each writer
# accepts ``(ds, file_path, **kwargs)`` and saves directly to disk.
_POSES_WRITERS: dict[str, Callable] = {
    "DeepLabCut": save_poses.to_dlc_file,
    "SLEAP": save_poses.to_sleap_analysis_file,
    "LightningPose": save_poses.to_lp_file,
}

# Mapping of bounding-boxes-format names to the writer that handles them.
_BBOXES_WRITERS: dict[str, Callable] = {
    "VIA-tracks": save_bboxes.to_via_tracks_file,
}

# The ``ds_type`` each non-netCDF target is compatible with.
_TARGET_DS_TYPE: dict[str, Literal["poses", "bboxes"]] = {
    **{name: "poses" for name in _POSES_WRITERS},
    "NWB": "poses",
    **{name: "bboxes" for name in _BBOXES_WRITERS},
}


def save_dataset(
    ds: xr.Dataset,
    file: str | Path,
    source_software: SaveTarget | None = None,
    **kwargs,
) -> None:
    """Save a ``movement`` dataset to a file in any supported format.

    This is a unified entry point for saving ``movement`` datasets, mirroring
    :func:`movement.io.load_dataset`. Based on the value of
    ``source_software``, the appropriate format-specific writer is called.

    Parameters
    ----------
    ds
        The ``movement`` poses or bounding boxes dataset to save.
    file
        Path to the file to save the dataset to. The required file extension
        depends on the target format (see the format-specific writers listed
        under "See Also").
    source_software
        The format to save the dataset in. If None (default), the dataset is
        saved in ``movement``'s native netCDF format (the file extension must
        be ``.nc``). Otherwise, one of the supported third-party formats:
        ``"DeepLabCut"``, ``"SLEAP"``, ``"LightningPose"``, ``"NWB"`` (poses
        only), or ``"VIA-tracks"`` (bounding boxes only). The value
        ``"netCDF"`` may also be passed explicitly.
    **kwargs
        Additional keyword arguments passed to the format-specific writer
        (e.g. ``split_individuals`` for DeepLabCut, ``config`` for NWB, or any
        :meth:`xarray.Dataset.to_netcdf` argument for netCDF).

    Raises
    ------
    TypeError
        If ``ds`` is not an :class:`xarray.Dataset`.
    ValueError
        If ``source_software`` is not a supported target, or if it is
        incompatible with the dataset's ``ds_type`` (e.g. saving a bounding
        boxes dataset to a poses-only format).

    See Also
    --------
    movement.io.save_poses.to_dlc_file
    movement.io.save_poses.to_sleap_analysis_file
    movement.io.save_poses.to_lp_file
    movement.io.save_poses.to_nwb_file
    movement.io.save_bboxes.to_via_tracks_file

    Notes
    -----
    For NWB, each individual is written to a separate file (as required by the
    NWB format). For a single-individual dataset, the dataset is written to
    ``file``. For a multi-individual dataset, the individual's name is appended
    to the file path, just before the extension, e.g.
    ``"/path/to/file_id_0.nwb"``.

    Writers for the Anipose format and changes to the DeepLabCut splitting
    default are not yet covered by this function (see issues #314 and #965).

    Examples
    --------
    Save a dataset to ``movement``'s native netCDF format:

    >>> from movement.io import save_dataset
    >>> save_dataset(ds, "/path/to/file.nc")

    Save a poses dataset to a DeepLabCut .h5 file:

    >>> save_dataset(ds, "/path/to/file.h5", source_software="DeepLabCut")

    Save a bounding boxes dataset to a VIA-tracks .csv file:

    >>> save_dataset(ds, "/path/to/file.csv", source_software="VIA-tracks")

    """
    if not isinstance(ds, xr.Dataset):
        raise logger.error(
            TypeError(f"Expected an xarray Dataset, but got {type(ds)}.")
        )

    target = source_software if source_software is not None else "netCDF"

    if target == "netCDF":
        _save_netcdf(ds, file, **kwargs)
        return

    if target not in _TARGET_DS_TYPE:
        supported = ", ".join(["netCDF", *_TARGET_DS_TYPE])
        raise logger.error(
            ValueError(
                f"Unsupported source_software for saving: '{target}'. "
                f"Supported values are: {supported}."
            )
        )

    _validate_ds_type(ds, target)

    if target == "NWB":
        _save_nwb(ds, file, **kwargs)
        return

    writer = {**_POSES_WRITERS, **_BBOXES_WRITERS}[target]
    writer(ds, file, **kwargs)


def _validate_ds_type(ds: xr.Dataset, target: str) -> None:
    """Check that the dataset's ``ds_type`` is compatible with the target.

    The check is skipped if the dataset has no ``ds_type`` attribute, in which
    case the format-specific writer's own validation will catch any mismatch.
    """
    expected = _TARGET_DS_TYPE[target]
    ds_type = ds.attrs.get("ds_type")
    if ds_type is not None and ds_type != expected:
        raise logger.error(
            ValueError(
                f"Cannot save a '{ds_type}' dataset to the '{target}' format, "
                f"which expects a '{expected}' dataset."
            )
        )


def _save_netcdf(ds: xr.Dataset, file: str | Path, **kwargs) -> None:
    """Save a ``movement`` dataset to a netCDF file."""
    valid_path = validate_file_path(file, permission="w", suffixes={".nc"})
    ds.to_netcdf(valid_path, **kwargs)
    logger.info(f"Saved dataset to {valid_path}.")


def _save_nwb(ds: xr.Dataset, file: str | Path, **kwargs) -> None:
    """Save a ``movement`` poses dataset to one or more NWB files.

    :func:`movement.io.save_poses.to_nwb_file` builds the NWBFile object(s)
    but does not write them to disk; this helper writes them. Multi-individual
    datasets yield one file per individual, with the individual name appended
    to the file path.
    """
    valid_path = validate_file_path(file, permission="w", suffixes={".nwb"})
    nwb_files = save_poses.to_nwb_file(ds, **kwargs)
    if isinstance(nwb_files, pynwb.file.NWBFile):
        _write_nwb_to_disk(nwb_files, valid_path)
    else:
        for nwb_file in nwb_files:
            individual_path = valid_path.with_name(
                f"{valid_path.stem}_{nwb_file.identifier}{valid_path.suffix}"
            )
            _write_nwb_to_disk(nwb_file, individual_path)


def _write_nwb_to_disk(nwb_file: pynwb.file.NWBFile, file_path: Path) -> None:
    """Write a single NWBFile object to disk."""
    with pynwb.NWBHDF5IO(file_path, mode="w") as io:
        io.write(nwb_file)
    logger.info(f"Saved dataset to {file_path}.")
