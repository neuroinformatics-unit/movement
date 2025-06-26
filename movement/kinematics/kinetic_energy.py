"""Functions for computing kinetic energy."""

import numpy as np
import xarray as xr

from movement.kinematics.kinematics import compute_velocity
from movement.utils.vector import compute_norm
from movement.validators.arrays import validate_dims_coords


def compute_kinetic_energy(
    position: xr.DataArray,
    keypoints: list | None = None,
    masses: dict | None = None,
) -> xr.DataArray:
    """Compute translational and rotational kinetic energy per individual.

    We consider each individual's set of keypoints (pose) as a classical
    system of bodies in physics. This function requires at least two
    keypoints per individual, but more would be desirable for a meaningful
    decomposition into translational and rotational components.

    Parameters
    ----------
    position : xr.DataArray
        The position data array with dimensions
        ('time', 'individuals', 'keypoints', 'space').

    keypoints : list, optional
        A list of keypoint names to include in the computation.
        By default, all are used.

    masses : dict, optional
        A dictionary mapping keypoint names to mass, e.g.
        {"snout": 1.2, "tail": 0.8}.
        By default, unit mass is assumed for all keypoints.

    Returns
    -------
    xr.DataArray
        A DataArray with dimensions ('time', 'individuals', 'energy'),
        where energy = ['translational', 'rotational'].

    Examples
    --------
    >>> from movement.kinematics.kinetic_energy import compute_kinetic_energy

    Compute translational and rotational kinetic energy from positions:

    >>> position = xr.DataArray(
    ...     np.random.rand(3, 2, 4, 2),
    ...     coords={
    ...         "time": [0, 1, 2],
    ...         "individuals": ["id0", "id1"],
    ...         "keypoints": ["snout", "spine", "tail_base", "tail_tip"],
    ...         "space": ["x", "y"],
    ...     },
    ...     dims=["time", "individuals", "keypoints", "space"],
    ... )
    >>> energy = compute_kinetic_energy(position)
    >>> energy
    <xarray.DataArray (time: 3, individuals: 2, energy: 2)>
    Coordinates:
      * time        (time) int64 0 1 2
      * individuals (individuals) <U3 'id0' 'id1'
      * energy      (energy) <U13 'translational' 'rotational'

    Compute total kinetic energy:
    >>> total_ke = energy.sum(dim="energy")

    Exclude an unreliable keypoint (e.g. "tail_tip"):
    >>> energy = compute_kinetic_energy(
    ...     position, keypoints=["snout", "spine", "tail_base"]
    ... )

    Use custom keypoint masses:
    >>> masses = {"snout": 1.2, "spine": 0.8, "tail_base": 1.0}
    >>> energy = compute_kinetic_energy(position, masses=masses)

    """
    # Validate required dimensions and coordinate labels
    validate_dims_coords(
        position, {"time": [], "space": ["x", "y"], "keypoints": []}
    )

    # Validate that at least 2 keypoints exist
    if keypoints is not None:
        if len(keypoints) < 2:
            raise ValueError(
                "At least two keypoints are required for KE decomposition."
            )
    elif position.sizes["keypoints"] < 2:
        raise ValueError("Position array must contain at least two keypoints.")

    # Subset keypoints if requested
    if keypoints is not None:
        position = position.sel(keypoints=keypoints)

    # Compute velocity from position
    velocity = compute_velocity(position)

    # Handle masses weights
    if masses:
        weights = xr.DataArray(
            [masses.get(str(k.item()), 1.0) for k in position.keypoints],
            dims=["keypoints"],
        )
    else:
        weights = xr.DataArray(
            np.ones(position.sizes["keypoints"]), dims=["keypoints"]
        )

    # Compute per-keypoint kinetic energy
    vel_squared = (
        compute_norm(velocity) ** 2
    )  # shape: (time, individuals, keypoints)
    weighted_ke = 0.5 * vel_squared * weights

    # Total kinetic energy
    K_total = weighted_ke.sum(dim="keypoints")

    # Compute center of mass velocity
    mass_sum = weights.sum()
    v_cm = (velocity * weights.expand_dims(space=["x", "y"])).sum(
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
