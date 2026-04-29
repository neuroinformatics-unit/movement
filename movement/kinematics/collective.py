"""Kinematics utilities for groups of individuals.

Currently contains functions to compute group-level spread metrics.
"""

import numpy as np
import xarray as xr


def compute_group_spread(
    position: xr.DataArray,
    *,
    keypoint: str = "centroid",
    method: str = "radius_of_gyration",
) -> xr.DataArray:
    """Compute per-frame group spread (radius of gyration) from positions.

    Parameters
    ----------
    position : xr.DataArray
        Positions with dims including ``time``, ``space`` and
        ``individuals``. May also include ``keypoints``.
    keypoint : str, default "centroid"
        Keypoint to use when ``keypoints`` coordinate exists.
    method : str, default "radius_of_gyration"
        Currently only "radius_of_gyration" is supported.

    Returns
    -------
    xr.DataArray
        Spread per frame (dimension: ``time``).

    """
    required_dims = {"time", "space", "individuals"}
    # mypy: position.dims is tuple[Hashable, ...]; convert to str for set ops
    missing_dims = required_dims - set(map(str, position.dims))
    if missing_dims:
        raise ValueError(
            f"`position` must contain dimensions {sorted(required_dims)}. "
            f"Missing: {sorted(missing_dims)}"
        )

    if "keypoints" in position.dims:
        kp_coord = position.coords.get("keypoints")
        if position.sizes.get("keypoints", 0) == 1:
            position = position.isel(keypoints=0)
        else:
            # try to select the requested keypoint (defaults to 'centroid')
            if kp_coord is not None and keypoint in list(kp_coord.values):
                position = position.sel(keypoints=keypoint)
            else:
                raise ValueError(
                    "Multiple keypoints present; pass `keypoint` to select "
                    "one or include a keypoint named 'centroid'."
                )

    # validate method
    if method != "radius_of_gyration":
        raise ValueError(
            f"Unsupported method '{method}'. Supported: 'radius_of_gyration'"
        )

    center = position.mean(dim="individuals", skipna=True)
    diff = position - center
    sqdist = (diff**2).sum(dim="space")
    rg2 = sqdist.mean(dim="individuals", skipna=True)

    spread: xr.DataArray = xr.apply_ufunc(np.sqrt, rg2)
    spread.name = "group_spread"
    spread.attrs["method"] = method

    return spread
