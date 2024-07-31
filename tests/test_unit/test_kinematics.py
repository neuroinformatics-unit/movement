from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.analysis import kinematics


@pytest.fixture
def expected_kinematic_arrays_pose_dataset(
    valid_poses_dataset,
):  # Compute by hand? using np.gradient?
    """Return a function to generate the expected dataarray
    for different kinematic properties.
    """

    def _expected_dataarray(property):
        """Return an xarray.DataArray with default values and
        the expected dimensions and coordinates.
        """
        # Expected x,y values for velocity
        x_vals = np.array(
            [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 17.0]
        )
        y_vals = np.full((10, 2, 2, 1), 4.0)
        if property == "acceleration":
            x_vals = np.array(
                [1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.0]
            )
            y_vals = np.full((10, 2, 2, 1), 0)
        elif property == "displacement":
            x_vals = np.array(
                [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]
            )
            y_vals[0] = 0

        x_vals = x_vals.reshape(-1, 1, 1, 1)
        # Repeat the x_vals to match the shape of the position
        x_vals = np.tile(x_vals, (1, 2, 2, 1))
        return xr.DataArray(
            np.concatenate(
                [x_vals, y_vals],
                axis=-1,
            ),
            dims=valid_poses_dataset.dims,
            coords=valid_poses_dataset.coords,
        )

    return _expected_dataarray


@pytest.mark.parametrize(
    "valid_dataset, expected_exception",
    [
        ("valid_poses_dataset", does_not_raise()),
        ("valid_poses_dataset_with_nan", does_not_raise()),
        ("missing_dim_poses_dataset", pytest.raises(AttributeError)),
        # ("valid_bboxes_dataset", does_not_raise()),
    ],
)
def test_displacement(
    valid_dataset,
    expected_exception,
    expected_kinematic_arrays_pose_dataset,
    request,
):
    """Test displacement computation."""
    input_dataset = request.getfixturevalue(valid_dataset)
    position = input_dataset.position

    with expected_exception:
        result = kinematics.compute_displacement(position)
        expected = expected_kinematic_arrays_pose_dataset("displacement")
        if input_dataset.position.isnull().any():
            expected.loc[{"individuals": "ind1", "time": [3, 4, 7, 8, 9]}] = (
                np.nan
            )
        xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "valid_dataset, expected_exception",
    [
        ("valid_poses_dataset", does_not_raise()),
        ("valid_poses_dataset_with_nan", does_not_raise()),
        ("missing_dim_poses_dataset", pytest.raises(AttributeError)),
        # ("valid_bboxes_dataset", does_not_raise()),
    ],
)
def test_velocity(
    valid_dataset,
    expected_exception,
    expected_kinematic_arrays_pose_dataset,
    request,
):
    """Test velocity computation."""
    valid_dataset = request.getfixturevalue(valid_dataset)
    with expected_exception:
        result = kinematics.compute_velocity(valid_dataset.position)
        expected = expected_kinematic_arrays_pose_dataset("velocity")
        if valid_dataset.position.isnull().any():
            expected.loc[
                {"individuals": "ind1", "time": [2, 4, 6, 7, 8, 9]}
            ] = np.nan
        xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "valid_dataset, expected_exception",
    [
        ("valid_poses_dataset", does_not_raise()),
        ("valid_poses_dataset_with_nan", does_not_raise()),
        ("missing_dim_poses_dataset", pytest.raises(AttributeError)),
        # ("valid_bboxes_dataset", does_not_raise()),
    ],
)
def test_acceleration(
    valid_dataset,
    expected_exception,
    expected_kinematic_arrays_pose_dataset,
    request,
):
    """Test acceleration computation."""
    valid_dataset = request.getfixturevalue(valid_dataset)
    with expected_exception:
        result = kinematics.compute_acceleration(valid_dataset.position)
        expected = expected_kinematic_arrays_pose_dataset("acceleration")
        if valid_dataset.position.isnull().any():
            expected.loc[
                {"individuals": "ind1", "time": [1, 3, 5, 6, 7, 8, 9]}
            ] = np.nan
        xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("order", [0, -1, 1.0, "1"])
def test_approximate_derivative_with_invalid_order(order):
    """Test that an error is raised when the order is non-positive."""
    data = np.arange(10)
    expected_exception = ValueError if isinstance(order, int) else TypeError
    with pytest.raises(expected_exception):
        kinematics._compute_approximate_derivative(data, order=order)
