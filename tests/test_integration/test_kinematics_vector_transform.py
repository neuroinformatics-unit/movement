import math

import numpy as np
import pytest
import xarray as xr

import movement.kinematics as kin
from movement.utils import vector


@pytest.mark.parametrize(
    "valid_dataset", ["valid_poses_dataset", "valid_bboxes_dataset"]
)
@pytest.mark.parametrize(
    "kinematic_variable, expected_kinematics_polar",
    [
        (
            "displacement",
            [
                np.vstack(
                    [
                        np.zeros((1, 2)),
                        np.tile([math.sqrt(2), math.atan(1)], (9, 1)),
                    ],
                ),  # Individual 0, rho=sqrt(2), phi=45deg
                np.vstack(
                    [
                        np.zeros((1, 2)),
                        np.tile([math.sqrt(2), -math.atan(1)], (9, 1)),
                    ]
                ),  # Individual 1, rho=sqrt(2), phi=-45deg
            ],
        ),
        (
            "velocity",
            [
                np.tile(
                    [math.sqrt(2), math.atan(1)], (10, 1)
                ),  # Individual O, rho, phi=45deg
                np.tile(
                    [math.sqrt(2), -math.atan(1)], (10, 1)
                ),  # Individual 1, rho, phi=-45deg
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
    kinematic_array_pol = vector.cart2pol(kinematic_array_cart)

    # Build expected data array
    expected_array_pol = xr.DataArray(
        np.stack(expected_kinematics_polar, axis=-1),
        # Stack along the "individuals" axis
        dims=["time", "space", "individuals"],
    )
    if "keypoints" in ds.position.coords:
        expected_array_pol = expected_array_pol.expand_dims(
            {"keypoints": ds.position.coords["keypoints"].size}
        )
        expected_array_pol = expected_array_pol.transpose(
            "time", "space", "keypoints", "individuals"
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
