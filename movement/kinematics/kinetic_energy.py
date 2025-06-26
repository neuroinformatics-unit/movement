"""Compute kinetic energy decomposition per individual.

This includes total, translational, and rotational kinetic energy,
as defined in issue #228. All individuals are assumed to have unit masses
unless a mass dictionary is provided.

Based on feature request by: @niksirbi
"""

import numpy as np
import xarray as xr

from movement.utils.vector import compute_norm


def compute_kinetic_energy(
    velocities: xr.DataArray,
    keypoints: list | None = None,
    masses: dict | None = None,
) -> xr.DataArray:
    """Compute kinetic energy per individual.

    This includes translational and rotational components.

    Parameters
    ----------
    velocities : xr.DataArray
        The velocity data array with dimensions
        ('time', 'individuals', 'keypoints', 'xy') or ('...', 'space').

    keypoints : list, optional
        A list of keypoint indices to include in the computation.

    masses : dict, optional
        A dictionary mapping keypoint indices to mass, e.g., {0: 1.2, 1: 0.8}.

    Returns
    -------
    xr.DataArray
        A DataArray with dimensions ('time', 'individuals', 'energy'),
        where energy = ['translational', 'rotational'].

    """
    # 🛠️ Make dimension compatible
    if "xy" in velocities.dims:
        velocities = velocities.rename({"xy": "space"}).assign_coords(
            space=["x", "y"]
        )
    elif "space" in velocities.dims:
        velocities = velocities.assign_coords(space=["x", "y"])
    else:
        raise ValueError(
            "Expected spatial dimension named either 'xy' or 'space'."
        )

    # Apply keypoint filtering
    if keypoints is not None:
        velocities = velocities.sel(keypoints=keypoints)

    # Handle masses
    if masses:
        weights = xr.DataArray(
            [masses.get(int(k.item()), 1.0) for k in velocities.keypoints],
            dims=["keypoints"],
        )
    else:
        weights = xr.DataArray(
            np.ones(velocities.sizes["keypoints"]), dims=["keypoints"]
        )

    # Compute per-keypoint kinetic energy
    vel_squared = (
        compute_norm(velocities) ** 2
    )  # shape: (time, individuals, keypoints)
    weighted_ke = 0.5 * vel_squared * weights

    # Total kinetic energy
    K_total = weighted_ke.sum(dim="keypoints")

    # Compute center of mass velocity
    mass_sum = weights.sum()
    v_cm = (velocities * weights.expand_dims(space=["x", "y"])).sum(
        dim="keypoints"
    ) / mass_sum

    # Compute translational KE
    K_trans = 0.5 * compute_norm(v_cm) ** 2  # shape: (time, individuals)

    # Rotational KE
    K_rot = K_total - K_trans

    # Format output
    energy = xr.concat([K_trans, K_rot], dim="energy")
    energy = energy.assign_coords(energy=["translational", "rotational"])
    energy = energy.transpose("time", "individuals", "energy")
    return energy
