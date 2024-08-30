import math

import numpy as np
import pytest
import xarray as xr

from movement.utils import vector


@pytest.mark.parametrize(
    "valid_dataset_uniform_linear_motion",
    [
        "valid_poses_dataset_uniform_linear_motion",
        "valid_bboxes_dataset",
    ],
)
@pytest.mark.parametrize(
    "kinematic_variable, expected_2D_pol_array_per_individual",
    [
        (
            "displacement",
            {
                0: np.vstack(
                    [
                        np.zeros((1, 2)),
                        np.tile(
                            np.array(
                                [math.sqrt(2), math.atan(1)]
                            ),  # rho, phi=45deg
                            (9, 1),
                        ),
                    ]
                ),
                1: np.vstack(
                    [
                        np.zeros((1, 2)),
                        np.tile(
                            np.array(
                                [math.sqrt(2), -math.atan(1)]
                            ),  # rho, phi=-45deg
                            (9, 1),
                        ),
                    ]
                ),
            },
        ),
        (
            "velocity",
            {
                0: np.tile(
                    np.array([math.sqrt(2), math.atan(1)]),  # rho, phi=-45deg
                    (10, 1),
                ),
                1: np.tile(
                    np.array(
                        [math.sqrt(2), -math.atan(1)]
                    ),  # rho, phi=-135deg
                    (10, 1),
                ),
            },
        ),
        (
            "acceleration",
            {
                0: np.zeros((10, 2)),
                1: np.zeros((10, 2)),
            },
        ),
    ],
)
def test_cart2pol_transform_on_kinematics(
    valid_dataset_uniform_linear_motion,
    kinematic_variable,
    expected_2D_pol_array_per_individual,
    request,
):
    """Test transformation between Cartesian and polar coordinates
    with various kinematic properties.
    """
    ds = request.getfixturevalue(valid_dataset_uniform_linear_motion)
    kinematic_array_cart = getattr(ds.move, f"compute_{kinematic_variable}")()

    kinematic_array_pol = vector.cart2pol(kinematic_array_cart)

    # Check the polar array is as expected
    for ind in expected_2D_pol_array_per_individual:
        if "keypoints" in ds.position.coords:
            for k in range(ds.position.coords["keypoints"].size):
                assert np.allclose(
                    kinematic_array_pol.isel(
                        individuals=ind, keypoints=k
                    ).values,
                    expected_2D_pol_array_per_individual[ind],
                )
        else:
            assert np.allclose(
                kinematic_array_pol.isel(individuals=ind).values,
                expected_2D_pol_array_per_individual[ind],
            )

    # Check we can recover the original Cartesian array?
    kinematic_array_cart_recover = vector.pol2cart(kinematic_array_pol)
    xr.testing.assert_allclose(
        kinematic_array_cart, kinematic_array_cart_recover
    )
