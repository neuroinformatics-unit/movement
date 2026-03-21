import numpy as np
import xarray as xr

from movement.kinematics.kinematics import compute_velocity
from movement.utils.logging import logger
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


def compute_kinetic_energy(
    position: xr.DataArray,
    keypoints: list | None = None,
    masses: dict | None = None,
    decompose: bool = False,
    normalize: bool = False,
    return_per_keypoint: bool = False,
) -> xr.DataArray:
    r"""Compute kinetic energy per individual.
    Enhanced version with:
    - normalization option
    - per-keypoint energy output
    - safer mass handling

    Parameters
    ----------
    position
        Input data with ``time``, ``space`` and ``keypoints``.
    keypoints
        Subset of keypoints to include.
    masses
        Dictionary mapping keypoints to masses.
    decompose
        If True, return translational and internal KE.
    normalize
        If True, divide KE by total mass (mass-invariant energy).
    return_per_keypoint
        If True, return KE per keypoint instead of summed.

    Returns
    -------
    xarray.DataArray

    """
    validate_dims_coords(
        position, {"time": [], "space": ["x", "y"], "keypoints": []}
    )
    if keypoints is not None:
        position = position.sel(keypoints=keypoints)
    if position.sizes["keypoints"] == 0:
        raise logger.error(ValueError("No keypoints available."))
    if decompose and position.sizes["keypoints"] < 2:
        raise logger.error(
            ValueError("At least 2 keypoints required for decomposition.")
        )
    velocity = compute_velocity(position)
    weights = xr.DataArray(
        np.ones(position.sizes["keypoints"]),
        dims=["keypoints"],
        coords={"keypoints": position.coords["keypoints"]},
    )
    if masses:
        for keypoint, mass in masses.items():
            if keypoint not in weights.coords["keypoints"]:
                raise logger.error(
                    ValueError(
                        f"Mass provided for unknown keypoint: {keypoint}"
                    )
                )
            if mass < 0:
                raise logger.error(
                    ValueError(f"Mass must be non-negative, got {mass}")
                )
            weights.loc[keypoint] = mass
    total_mass = weights.sum()
    ke_per_keypoint = 0.5 * weights * (compute_norm(velocity) ** 2)
    if return_per_keypoint:
        result = ke_per_keypoint
        result.name = "kinetic_energy_per_keypoint"
        return result
    ke_total = ke_per_keypoint.sum(dim="keypoints")
    if normalize:
        ke_total = ke_total / total_mass
    if not decompose:
        ke_total.name = "kinetic_energy"
        return ke_total
    weights_expanded = weights.expand_dims(space=["x", "y"])
    v_cm = (velocity * weights_expanded).sum(dim="keypoints") / total_mass
    ke_trans = 0.5 * total_mass * (compute_norm(v_cm) ** 2)
    if normalize:
        ke_trans = ke_trans / total_mass
    ke_int = ke_total - ke_trans
    ke = xr.concat([ke_trans, ke_int], dim="energy")
    ke = ke.assign_coords(energy=["translational", "internal"])
    ke = ke.transpose("time", ..., "energy")
    ke.name = "kinetic_energy"

    return ke
