import math

import numpy as np
import pytest
import xarray as xr

import movement.kinematics as kin
from movement.utils import vector

# Displacement vectors in polar coordinates
# for individual 0, with 10 time points and 2 space dimensions
# moving along x = y in the x positive, y positive direction

# forward displacement (rho = √2, phi = π/4)
forward_displacement_polar = np.vstack(
    [
        np.tile([math.sqrt(2), math.pi / 4], (9, 1)),
        np.zeros((1, 2)),
        # at time t=10, the forward displacement Cartesian vector is (x=0,y=0)
    ]
)


@pytest.mark.parametrize(
    "valid_dataset", ["valid_poses_dataset", "valid_bboxes_dataset"]
)
@pytest.mark.parametrize(
    "kinematic_variable, expected_kinematics_polar",
    [
        (
            "forward_displacement",
            [
                forward_displacement_polar,
                # Individual 0, rho = √2, phi = 45deg = π/4
                forward_displacement_polar * np.array([[1, -1]]),
                # Individual 1, rho = √2, phi = -45deg = -π/4
            ],
        ),
        (
            "backward_displacement",
            [
                np.roll(
                    forward_displacement_polar * np.array([[1, -3]]),
                    shift=1,
                    axis=0,
                ),
                # Individual 0, rho = √2, phi = -135deg = -3π/4
                np.roll(
                    forward_displacement_polar * np.array([[1, 3]]),
                    shift=1,
                    axis=0,
                ),
                # Individual 1, rho = √2, phi = 135deg = 3π/4
            ],
        ),
        (
            "velocity",
            [
                np.tile(
                    [math.sqrt(2), math.pi / 4], (10, 1)
                ),  # Individual O, rho=√2, phi=45deg=π/4
                np.tile(
                    [math.sqrt(2), -math.pi / 4], (10, 1)
                ),  # Individual 1, rho=√2, phi=-45deg=-π/4
            ],
        ),
        (
            "acceleration",
            [
                np.zeros((10, 2)),  # Individual 0
                np.zeros((10, 2)),  # Individual 1
            ],
        ),
    ],
)
def test_cart2pol_transform_on_kinematics(
    valid_dataset, kinematic_variable, expected_kinematics_polar, request
):
    """Test transformation between Cartesian and polar coordinates
    with various kinematic properties.
    """
    ds = request.getfixturevalue(valid_dataset)
    kinematic_array_cart = getattr(kin, f"compute_{kinematic_variable}")(
        ds.position
    )
    assert kinematic_array_cart.name == kinematic_variable
    kinematic_array_pol = vector.cart2pol(kinematic_array_cart)

    # Build expected data array
    expected_array_pol = xr.DataArray(
        np.stack(expected_kinematics_polar, axis=-1),
        # Stack along the "individuals" axis
        dims=["time", "space", "individual"],
    )
    if "keypoint" in ds.position.coords:
        expected_array_pol = expected_array_pol.expand_dims(
            {"keypoint": ds.position.coords["keypoint"].size}
        )
        expected_array_pol = expected_array_pol.transpose(
            "time", "space", "keypoint", "individual"
        )

    # Compare the values of the kinematic_array against the expected_array
    np.testing.assert_allclose(
        kinematic_array_pol.values, expected_array_pol.values
    )

    # Check we can recover the original Cartesian array
    kinematic_array_cart_recover = vector.pol2cart(kinematic_array_pol)
    xr.testing.assert_allclose(
        kinematic_array_cart, kinematic_array_cart_recover
    )
