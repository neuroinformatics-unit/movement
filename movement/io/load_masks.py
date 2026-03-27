"""Load segmentation masks and associated tracking data into ``movement``."""

from pathlib import Path

import dask.array as da
import pandas as pd
import xarray as xr

from movement.utils.logging import logger


def load_octron_bboxes(
    file_path: Path | str,
    extra_data_vars: bool = False,
    fps: float | None = None,
) -> xr.Dataset:
    """Load bounding box data from an OCTRON CSV file.

    Parameters
    ----------
    file_path : Path | str
        Path to the OCTRON CSV file.
    extra_data_vars : bool, optional
        If True, loads additional metrics (e.g., eccentricity, solidity,
        orientation) as extra data variables in the Dataset. Default is False.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers. If provided,
        the ``time`` coordinates will be converted to seconds.

    Returns
    -------
    xarray.Dataset
        An xarray Dataset containing the tracking data.

    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise logger.error(FileNotFoundError(f"File not found: {file_path}"))

    df = pd.read_csv(file_path)

    # Convert time coordinates if fps is provided
    time_coords = df.index.values
    if fps is not None:
        time_coords = time_coords / fps

    # Extract core bounding box coordinates
    # Note: Assuming standard 'x', 'y', 'width', 'height' columns exist
    core_vars = {
        "bboxes": (
            ["time", "individuals", "features"],
            df[["x", "y", "width", "height"]].values.reshape(len(df), 1, 4),
        )
    }

    ds = xr.Dataset(
        data_vars=core_vars,
        coords={
            "time": time_coords,
            "individuals": ["ind_0"],  # Placeholder for single individual
            "features": ["x", "y", "width", "height"],
        },
    )

    # Optimization: Load heavy extra metrics only if explicitly requested
    if extra_data_vars:
        for extra_col in ["eccentricity", "solidity", "orientation"]:
            if extra_col in df.columns:
                ds[extra_col] = (
                    ["time", "individuals"],
                    df[[extra_col]].values,
                )

    return ds


def load_masks_from_zarr(
    zarr_paths: dict[str, Path | str],
    chunk_size: tuple[int, int, int] = (100, -1, -1),
) -> xr.DataArray:
    """Lazily load instance segmentation masks from Zarr files.

    Ensures a lossless, memory-efficient boolean representation using Dask.

    Parameters
    ----------
    zarr_paths : dict[str, Path | str]
        A dictionary mapping individual names to their Zarr file paths.
        Example: {"ind_0": "path/to/mask1.zarr"}
    chunk_size : tuple of int, optional
        Chunking strategy for Dask. Default chunks across time (100 frames),
        keeping full spatial dimensions (-1, -1).

    Returns
    -------
    xarray.DataArray
        A lazily evaluated DataArray with dimensions
        (time, individuals, x, y).

    """
    dask_arrays = []
    individuals = list(zarr_paths.keys())

    for ind in individuals:
        path = Path(zarr_paths[ind])
        if not path.exists():
            raise logger.error(FileNotFoundError(f"Zarr not found: {path}"))

        # Lazily reference the Zarr array, cast to bool to minimize memory
        # Assuming the mask is stored under the default root
        # or a specific array
        arr = da.from_zarr(str(path)).astype(bool)
        arr = arr.rechunk(chunk_size)
        dask_arrays.append(arr)

    # Stack individual mask arrays along a new 'individuals' axis (axis 1)
    stacked_masks = da.stack(dask_arrays, axis=1)

    mask_da = xr.DataArray(
        stacked_masks,
        dims=["time", "individuals", "x", "y"],
        coords={"individuals": individuals},
        name="segmentation_masks",
    )

    return mask_da
