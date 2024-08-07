import numpy as np
import pytest
import xarray as xr

from movement.analysis import kinematics


def _compute_expected_displacement(position):
    displacement = position.diff("time")
    displacement = displacement.reindex_like(position)

    # set first frame to displacement 0
    displacement.loc[{"time": 0}] = 0
    return displacement


def _compute_expected_velocity(position):
    # differentiate position along time dimension
    return position.differentiate("time")


def _compute_expected_acceleration(position):
    velocity = _compute_expected_velocity(position)
    return velocity.differentiate("time")


@pytest.mark.parametrize(
    "valid_dataset",
    [
        "valid_poses_dataset",
        "valid_poses_dataset_with_nan",
        "valid_bboxes_dataset",
        "valid_bboxes_dataset_with_nan",
    ],
)
@pytest.mark.parametrize(
    "kinematic_variable, expected_kinematic_variable_fn",
    [
        ("displacement", _compute_expected_displacement),
        ("velocity", _compute_expected_velocity),
        ("acceleration", _compute_expected_acceleration),
    ],
)
def test_kinematics(
    valid_dataset,
    kinematic_variable,
    expected_kinematic_variable_fn,
    request,
):
    """Test displacement computation."""
    position = request.getfixturevalue(valid_dataset).position

    kinematic_variable = getattr(kinematics, f"compute_{kinematic_variable}")(
        position
    )
    expected_kinematic_variable = expected_kinematic_variable_fn(position)

    xr.testing.assert_allclose(kinematic_variable, expected_kinematic_variable)


@pytest.mark.parametrize(
    "invalid_dataset, expected_exception",
    [
        ("not_a_dataset", pytest.raises(AttributeError)),
        ("empty_dataset", pytest.raises(AttributeError)),
        ("missing_var_poses_dataset", pytest.raises(AttributeError)),
        ("missing_var_bboxes_dataset", pytest.raises(AttributeError)),
        ("missing_dim_poses_dataset", pytest.raises(AttributeError)),
        ("missing_dim_bboxes_dataset", pytest.raises(AttributeError)),
    ],
)
@pytest.mark.parametrize(
    "kinematic_variable",
    [
        "displacement",
        "velocity",
        "acceleration",
    ],
)
def test_kinematics_with_invalid_dataset(
    invalid_dataset,
    expected_exception,
    kinematic_variable,
    request,
):
    """Test displacement computation with an invalid dataset."""
    with expected_exception:
        position = request.getfixturevalue(invalid_dataset).position
        getattr(kinematics, f"compute_{kinematic_variable}")(position)


@pytest.mark.parametrize("order", [0, -1, 1.0, "1"])
def test_approximate_derivative_with_invalid_order(order):
    """Test that an error is raised when the order is non-positive."""
    data = np.arange(10)
    expected_exception = ValueError if isinstance(order, int) else TypeError
    with pytest.raises(expected_exception):
        kinematics._compute_approximate_time_derivative(data, order=order)
