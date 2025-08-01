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
    r"""Compute translational and internal kinetic energy per individual.

    We consider each individual's set of keypoints (pose) as a classical
    system of bodies in physics. This function requires at least two
    keypoints per individual, but more would be desirable for a meaningful
    decomposition into translational and internal components.

    Parameters
    ----------
    position : xr.DataArray
        The input data containing position information, with ``time``,
        ``space`` and ``keypoints`` as required dimensions.

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
        A data array containing the kinetic energy per individual, for every
        time point. Note that the output array lacks ``space`` and
        ``keypoints`` dimensions, but has an extra ``energy`` dimension
        with coordinates ``translational`` and ``internal``.

    Notes
    -----
    The total kinetic energy :math:`K_{total}` of a given individual
    at a single time point :math:`t` is given by:

    .. math::  K_{total} = \sum_{i} \frac{1}{2} m_i \| \mathbf{v}_i(t) \|^2

    where :math:`m_i` is the mass of the :math:`i`-th keypoint and
    :math:`\mathbf{v}_i(t)` is its velocity at time :math:`t`.

    Translational kinetic energy is computed based on the motion of the
    individual's center of mass:

    .. math:: K_{trans} = \frac{1}{2} M \| \mathbf{v}_{cm}(t) \|^2

    where :math:`M = \sum_{i} m_i` is the total mass of the individual
    and :math:`\mathbf{v}_{cm}(t) = \frac{1}{M} \sum_{i} m_i \mathbf{v}_i(t)`
    is the velocity of the center of mass at time :math:`t`.

    We define internal kinetic energy :math:`K_{int}` as the difference
    between the total kinetic energy and the translational kinetic energy:

    .. math:: K_{int} = K_{total} - K_{trans}

    This means that internal kinetic energy captures the motion of
    keypoints relative to the individual's center of mass.

    Examples
    --------
    >>> from movement.kinematics import compute_kinetic_energy

    Compute translational and internal kinetic energy from positions:

    >>> position = xr.DataArray(
    ...     np.random.rand(3, 2, 4, 2),
    ...     coords={
    ...         "time": [0, 1, 2],
    ...         "space": ["x", "y"],
    ...         "keypoints": ["snout", "spine", "tail_base", "tail_tip"],
    ...         "individuals": ["id0", "id1"],
    ...     },
    ...     dims=["time", "space", "keypoints", "individuals"],
    ... )
    >>> kinetic_energy = compute_kinetic_energy(position)
    >>> kinetic_energy
    <xarray.DataArray (time: 3, individuals: 2, energy: 2)>
    Coordinates:
      * time        (time) int64 0 1 2
      * individuals (individuals) <U3 'id0' 'id1'
      * energy      (energy) <U13 'translational' 'internal'

    Compute total kinetic energy:

    >>> total_ke = kinetic_energy.sum(dim="energy")

    Exclude an unreliable keypoint (e.g. "tail_tip"):

    >>> kinetic_energy = compute_kinetic_energy(
    ...     position, keypoints=["snout", "spine", "tail_base"]
    ... )

    Use unequal keypoint masses:

    >>> masses = {"snout": 1.2, "spine": 0.8, "tail_base": 1.0}
    >>> kinetic_energy = compute_kinetic_energy(position, masses=masses)

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

    # Compute total KE
    weighted_ke = 0.5 * weights * (compute_norm(velocity) ** 2)
    ke_total = weighted_ke.sum(dim="keypoints")

    # Compute translational KE based on center of mass velocity
    v_cm = (velocity * weights.expand_dims(space=["x", "y"])).sum(
        dim="keypoints"
    ) / weights.sum()
    ke_trans = 0.5 * weights.sum() * compute_norm(v_cm) ** 2

    # Rotational KE
    ke_int = ke_total - ke_trans

    # Format output
    ke = xr.concat([ke_trans, ke_int], dim="energy")
    ke = ke.assign_coords(energy=["translational", "internal"])
    ke = ke.transpose("time", ..., "energy")
    return ke
