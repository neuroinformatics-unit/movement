import numpy as np
import xarray as xr

from movement.kinematics.kinetic_energy import compute_kinetic_energy


def test_kinetic_energy_basic():
    # Simulated velocity data for 1 individual, 3 keypoints,
    # over 2 time steps
    velocities = xr.DataArray(
        np.array(
            [
                [  # time 0
                    [[1, 0], [0, 1], [1, 1]],  # individual 0
                ],
                [  # time 1
                    [[2, 0], [0, 2], [2, 2]],
                ],
            ]
        ),
        dims=["time", "individuals", "keypoints", "space"],
        coords={
            "time": [0, 1],
            "individuals": [0],
            "keypoints": [0, 1, 2],
            "space": ["x", "y"],
        },
    )

    result = compute_kinetic_energy(velocities)

    # Check dimensions and coordinate names
    assert set(result.dims) == {"time", "individuals", "energy"}
    assert list(result.coords["energy"].values) == [
        "translational",
        "rotational",
    ]

    # Basic shape and positivity checks
    assert result.shape == (2, 1, 2)
    assert (result >= 0).all()
