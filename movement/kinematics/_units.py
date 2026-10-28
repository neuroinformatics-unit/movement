"""Resolve explicitly declared, simple unit labels for kinematics outputs.

This is deliberately not a unit parser: compound or unknown labels are not
interpreted, and values are never converted between units.
"""

import xarray as xr

_LENGTH_UNITS = {
    "m",
    "cm",
    "mm",
    "um",
    "µm",
    "nm",
    "km",
    "px",
    "pixel",
    "pixels",
}
_ANGLE_UNITS = {"rad", "radians", "deg", "degrees"}
_TIME_UNITS = {
    "s": "s",
    "second": "s",
    "seconds": "s",
    "frame": "frame",
    "frames": "frame",
}


def _get_unit(data: xr.DataArray, *, spatial: bool = False) -> str | None:
    """Read a supported label, rejecting conflicting spatial declarations."""
    units = data.attrs.get("units")
    space_unit = data.attrs.get("space_unit")
    # Angles can retain spatial calibration metadata from their source.
    if not spatial and isinstance(units, str) and units in _ANGLE_UNITS:
        return units
    if any(
        value is not None and not isinstance(value, str)
        for value in (units, space_unit)
    ):
        return None
    if units is not None and space_unit is not None and units != space_unit:
        return None
    unit = units if units is not None else space_unit
    supported = _LENGTH_UNITS if spatial else _LENGTH_UNITS | _ANGLE_UNITS
    return unit if isinstance(unit, str) and unit in supported else None


def _get_time_unit(data: xr.DataArray) -> str | None:
    """Read explicit time metadata for numeric coordinates, without guessing.

    Dataset ``time_unit`` metadata is only available here if the caller
    copied it to the DataArray. Datetime differentiation uses xarray's own
    datetime unit, so coordinate labels must not be used to infer its scale.
    """
    if data.time.dtype.kind not in "iuf":
        return None
    declarations = [
        value
        for value in (
            data.time.attrs.get("units"),
            data.attrs.get("time_unit"),
        )
        if value is not None
    ]
    if not declarations:
        return None
    units = [
        _TIME_UNITS.get(value) if isinstance(value, str) else None
        for value in declarations
    ]
    return units[0] if units[0] is not None and len(set(units)) == 1 else None
