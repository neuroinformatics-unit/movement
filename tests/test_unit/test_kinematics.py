import numpy as np
import pytest

from movement.analysis import kinematics


@pytest.mark.parametrize(
    "valid_dataset_uniform_linear_motion",
    [
        "valid_poses_dataset_uniform_linear_motion",
        "valid_bboxes_dataset",
    ],
)
@pytest.mark.parametrize(
    "kinematic_variable, expected_2D_array_per_individual_and_kpt",
    [
        (
            "displacement",
            {
                0: np.vstack(  # first individual
                    [np.zeros((1, 2)), np.ones((9, 2))]
                ),  # at t=0 displacement is (0,0)
                1: np.multiply(
                    np.vstack([np.zeros((1, 2)), np.ones((9, 2))]),
                    np.array([1, -1]),
                ),  # n_frames, n_space_dims
            },
        ),
        (
            "velocity",
            {
                0: np.ones((10, 2)),
                1: np.multiply(np.ones((10, 2)), np.array([1, -1])),
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
def test_kinematics_uniform_linear_motion(
    valid_dataset_uniform_linear_motion,
    kinematic_variable,
    expected_2D_array_per_individual_and_kpt,
    request,
):
    """Test kinematics values for uniform linear motion case.

    Uniform linear motion means individuals move along a line
    at constant velocity.

    We consider 2 individuals ("id_0" and "id_1"),
    tracked for 10 frames, along x and y:
    - id_0 moves along x=y line from the origin
    - id_1 moves along x=-y line from the origin
    - they both move one unit (pixel) along each axis in each frame

    If the dataset is a poses dataset, we consider 3 keypoints per individual
    (centroid, left, right), that are always in front of the centroid at 45deg.
    """
    position = request.getfixturevalue(
        valid_dataset_uniform_linear_motion
    ).position
    kinematic_variable = getattr(kinematics, f"compute_{kinematic_variable}")(
        position
    )

    for ind in expected_2D_array_per_individual_and_kpt:
        if "keypoints" in position.coords:
            for k in range(position.coords["keypoints"].size):
                assert np.allclose(
                    kinematic_variable.isel(
                        individuals=ind, keypoints=k
                    ).values,
                    expected_2D_array_per_individual_and_kpt[ind],
                )
        else:
            assert np.allclose(
                kinematic_variable.isel(individuals=ind).values,
                expected_2D_array_per_individual_and_kpt[ind],
            )


@pytest.mark.parametrize(
    "invalid_dataset, expected_exception",
    [
        ("not_a_dataset", pytest.raises(AttributeError)),
        ("empty_dataset", pytest.raises(AttributeError)),
        ("missing_var_poses_dataset", pytest.raises(AttributeError)),
        ("missing_var_bboxes_dataset", pytest.raises(AttributeError)),
        ("missing_dim_poses_dataset", pytest.raises(ValueError)),
        ("missing_dim_bboxes_dataset", pytest.raises(ValueError)),
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
