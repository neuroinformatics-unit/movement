"""Load movement datasets from zarr format."""

from pathlib import Path

import xarray as xr

from movement.utils.logging import logger


def from_zarr(
    store: str | Path,
    *,
    chunks: str | dict | None = "auto",
    **kwargs,
) -> xr.Dataset:
    """Load a movement dataset from a zarr store.

    This uses xarray's built-in :func:`xarray.open_zarr` function to
    read the dataset, restoring all variables, coordinates, and metadata
    attributes.

    Parameters
    ----------
    store : str or pathlib.Path
        Path to the zarr store (directory).
    chunks : str, dict, or None, optional
        Chunk sizes for dask arrays. ``"auto"`` (default) uses dask's
        automatic chunking. ``None`` loads data eagerly into memory.
        A dict can specify per-dimension chunk sizes, e.g.
        ``{"time": 1000}``.
    **kwargs
        Additional keyword arguments passed to
        :func:`xarray.open_zarr`.

    Returns
    -------
    xarray.Dataset
        A movement dataset loaded from the zarr store.

    Raises
    ------
    FileNotFoundError
        If the zarr store directory does not exist.
    ValueError
        If the given path points to a file rather than a directory.

    See Also
    --------
    movement.io.save_zarr.to_zarr : Save a movement dataset to zarr.

    Examples
    --------
    Load a poses dataset from zarr:

    >>> from movement.io import load_zarr  # doctest: +SKIP
    >>> ds = load_zarr.from_zarr("/path/to/poses.zarr")  # doctest: +SKIP

    Load data eagerly into memory (no dask):

    >>> ds = load_zarr.from_zarr(  # doctest: +SKIP
    ...     "/path/to/poses.zarr", chunks=None
    ... )

    """
    store = Path(store)
    if not store.exists():
        raise logger.error(FileNotFoundError(f"Zarr store not found: {store}"))
    if not store.is_dir():
        raise logger.error(
            ValueError(
                f"Expected a directory (zarr store), got a file: {store}"
            )
        )
    logger.info(f"Loading dataset from zarr store: {store}")
    ds = xr.open_zarr(store=store, chunks=chunks, **kwargs)
    logger.info(f"Dataset loaded successfully from: {store}")
    return ds
