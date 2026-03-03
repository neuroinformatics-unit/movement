"""Save movement datasets to zarr format."""

from pathlib import Path
from typing import Literal

import xarray as xr

from movement.utils.logging import logger

ZarrWriteMode = Literal["w", "w-", "a", "a-", "r+", "r"]


def to_zarr(
    ds: xr.Dataset,
    store: str | Path,
    *,
    mode: ZarrWriteMode = "w",
    **kwargs,
) -> None:
    """Save a movement dataset to a zarr store.

    This uses xarray's built-in :meth:`xarray.Dataset.to_zarr` method
    to write the dataset, preserving all variables, coordinates, and
    metadata attributes (such as ``source_software``, ``source_file``,
    and ``fps``).

    Parameters
    ----------
    ds : xarray.Dataset
        A movement poses or bounding boxes dataset.
    store : str or pathlib.Path
        Path to the output zarr store (directory). The ``.zarr``
        suffix is conventional but not required.
    mode : str, optional
        Write mode. ``"w"`` (default) creates a new store or overwrites
        an existing one. ``"a"`` appends to an existing store.
        See :meth:`xarray.Dataset.to_zarr` for all supported modes.
    **kwargs
        Additional keyword arguments passed to
        :meth:`xarray.Dataset.to_zarr`.

    See Also
    --------
    movement.io.load_zarr.from_zarr : Load a movement dataset from zarr.

    Examples
    --------
    Save a poses dataset to zarr:

    >>> from movement.io import save_zarr  # doctest: +SKIP
    >>> save_zarr.to_zarr(ds, "/path/to/poses.zarr")  # doctest: +SKIP

    """
    store = Path(store)
    logger.info(f"Saving dataset to zarr store: {store}")
    ds.to_zarr(store=store, mode=mode, **kwargs)
    logger.info(f"Dataset saved successfully to: {store}")
