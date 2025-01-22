"""Transforms module."""

import xarray as xr


def scale(
    data_array: xr.DataArray, factor: float = 1.0, unit: str | None = None
) -> xr.DataArray:
    """Scale data by a given factor with an optional unit."""
    scaled_data_array = data_array * factor
    if unit is not None:
        scaled_data_array.attrs["unit"] = unit
    elif unit is None:
        scaled_data_array.attrs.pop("unit", None)
    return scaled_data_array
