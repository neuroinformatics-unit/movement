"""Functions for computing kinetic energy."""

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
) -> xr.DataArray:
    r"""Compute kinetic energy per individual.

    We consider each individual's set of keypoints (pose) as a classical
    system of particles in physics (see Notes).

    Parameters
    ----------
    position : xr.DataArray
        The input data containing position information, with ``time``,
        ``space`` and ``keypoint`` as required dimensions.
    keypoints : list, optional
        A list of keypoint names to include in the computation.
        By default, all are used.
    masses : dict, optional
        A dictionary mapping keypoint names to masses, e.g.
        {"snout": 1.2, "tail": 0.8}.
        By default, unit mass is assumed for all keypoints.
    decompose : bool, optional
        If True, the kinetic energy is decomposed into "translational" and
        "internal" components (see Notes). This requires at least two keypoints
        per individual, but more would be desirable for a meaningful
        decomposition. The default is False, meaning the total kinetic energy
        is returned.

    Returns
    -------
    xr.DataArray
        A data array containing the kinetic energy per individual, for every
        time point. Note that the output array lacks ``space`` and
        ``keypoint`` dimensions.
        If ``decompose=True`` an extra ``energy`` dimension is added,
        with coordinates ``translational`` and ``internal``.

    Notes
    -----
    Considering a given individual at time point :math:`t` as a system of
    keypoint particles, its total kinetic energy :math:`T_{total}` is given by:

    .. math::  T_{total} = \sum_{i} \frac{1}{2} m_i \| \mathbf{v}_i(t) \|^2

    where :math:`m_i` is the mass of the :math:`i`-th keypoint and
    :math:`\mathbf{v}_i(t)` is its velocity at time :math:`t`.

    From Samuel König's second theorem, we can decompose :math:`T_{total}`
    into:

    - Translational kinetic energy: the kinetic energy of the individual's
      total mass :math:`M` moving with the centre of mass velocity;
    - Internal kinetic energy: the kinetic energy of the keypoints moving
      relative to the individual's centre of mass.

    We compute translational kinetic energy :math:`T_{trans}` as follows:

    .. math:: T_{trans} = \frac{1}{2} M \| \mathbf{v}_{cm}(t) \|^2

    where :math:`M = \sum_{i} m_i` is the total mass of the individual
    and :math:`\mathbf{v}_{cm}(t) = \frac{1}{M} \sum_{i} m_i \mathbf{v}_i(t)`
    is the velocity of the centre of mass at time :math:`t`
    (computed as the weighted mean of keypoint velocities).

    Internal kinetic energy :math:`T_{int}` is derived as the difference
    between the total and translational components:

    .. math:: T_{int} = T_{total} - T_{trans}

    Examples
    --------
    >>> from movement.kinematics import compute_kinetic_energy
    >>> import numpy as np
    >>> import xarray as xr

    Compute total kinetic energy:

    >>> position = xr.DataArray(
    ...     np.random.rand(3, 2, 4, 2),
    ...     coords={
    ...         "time": np.arange(3),
    ...         "individuals": ["id0", "id1"],
    ...         "keypoint": ["snout", "spine", "tail_base", "tail_tip"],
    ...         "space": ["x", "y"],
    ...     },
    ...     dims=["time", "individuals", "keypoint", "space"],
    ... )

    >>> kinetic_energy_total = compute_kinetic_energy(position)

    >>> kinetic_energy_total
    <xarray.DataArray (time: 3, individuals: 2)> Size: 48B
    0.6579 0.7394 0.1304 0.05152 0.2436 0.5719
    Coordinates:
    * time         (time) int64 24B 0 1 2
    * individuals  (individuals) <U3 24B 'id0' 'id1'

    Compute kinetic energy decomposed into translational
    and internal components:

    >>> kinetic_energy = compute_kinetic_energy(position, decompose=True)

    >>> kinetic_energy
    <xarray.DataArray (time: 3, individuals: 2, energy: 2)> Size: 96B
    0.0172 1.318 0.02069 0.6498 0.02933 ... 0.1716 0.07829 0.7942 0.06901 0.857
    Coordinates:
    * time         (time) int64 24B 0 1 2
    * individuals  (individuals) <U3 24B 'id0' 'id1'
    * energy       (energy) <U13 104B 'translational' 'internal'

    Select the 'internal' component:

    >>> kinetic_energy_internal = kinetic_energy.sel(energy="internal")

    Use unequal keypoint masses and exclude an unreliable keypoint
    (e.g. "tail_tip"):

    >>> masses = {"snout": 1.2, "spine": 0.8, "tail_base": 1.0}

    >>> kinetic_energy = compute_kinetic_energy(
    ...     position,
    ...     keypoints=["snout", "spine", "tail_base"],
    ...     masses=masses,
    ...     decompose=True,
    ... )

    """
    # Validate required dimensions and coordinate labels
    validate_dims_coords(
        position, {"time": [], "space": ["x", "y"], "keypoint": []}
    )

    # Subset keypoints if requested
    if keypoints is not None:
        position = position.sel(keypoint=keypoints)

    # Validate that at least 2 keypoints exist for decomposition
    if decompose and position.sizes["keypoint"] < 2:
        raise logger.error(
            ValueError(
                "At least 2 keypoints are required to decompose "
                "kinetic energy into translational and internal components."
            )
        )

    # Compute velocity from position
    velocity = compute_velocity(position)

    # Initialise unit weights
    weights = xr.DataArray(
        np.ones(position.sizes["keypoint"]),
        dims=["keypoint"],
        coords={"keypoint": position.coords["keypoint"]},
    )

    # Update weights with keypoint masses, if provided
    if masses:
        for keypoint, mass in masses.items():
            weights.loc[keypoint] = mass

    # Compute total KE
    weighted_ke = 0.5 * weights * (compute_norm(velocity) ** 2)
    ke_total = weighted_ke.sum(dim="keypoint")

    if not decompose:
        return ke_total
    else:
        # Compute translational KE based on centre of mass velocity
        v_cm = (velocity * weights.expand_dims(space=["x", "y"])).sum(
            dim="keypoint"
        ) / weights.sum()
        ke_trans = 0.5 * weights.sum() * compute_norm(v_cm) ** 2

        # Internal KE
        ke_int = ke_total - ke_trans

        # Format output
        ke = xr.concat([ke_trans, ke_int], dim="energy")
        ke = ke.assign_coords(energy=["translational", "internal"])
        ke = ke.transpose("time", ..., "energy")
        return ke
