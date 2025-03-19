import numpy as np
import pytest
import xarray as xr

from movement.kinematics.kinematics import (
    compute_acceleration,
    compute_displacement,
    compute_path_length,
    compute_speed,
    compute_time_derivative,
    compute_velocity,
)


class TestComputeKinematics:
    """Test suite for computing kinematic variables."""

    @pytest.mark.parametrize(
        "valid_dataset", ["valid_poses_dataset", "valid_bboxes_dataset"]
    )
    @pytest.mark.parametrize(
        "kinematic_variable",
        ["displacement", "velocity", "acceleration", "speed", "path_length"],
    )
    def test_kinematics(self, valid_dataset, kinematic_variable, request):
        """Test kinematic computations with valid datasets."""
        position = request.getfixturevalue(valid_dataset).position
        kinematic_func = {
            "displacement": compute_displacement,
            "velocity": compute_velocity,
            "acceleration": compute_acceleration,
            "speed": compute_speed,
            "path_length": compute_path_length,
        }[kinematic_variable]
        kinematic_array = kinematic_func(position)
        assert isinstance(kinematic_array, xr.DataArray)
        if kinematic_variable == "speed":
            # Speed removes 'space' dim
            expected_dims = tuple(d for d in position.dims if d != "space")
            assert kinematic_array.dims == expected_dims, (
                f"Expected dims {expected_dims}, got {kinematic_array.dims}"
            )
        elif kinematic_variable != "path_length":
            # Displacement, velocity, acceleration retain 'space'
            assert "space" in kinematic_array.dims, (
                f"Expected 'space' in dims, got {kinematic_array.dims}"
            )
        else:
            # Path length sums over 'time' and 'space'
            assert (
                "time" not in kinematic_array.dims
                and "space" not in kinematic_array.dims
            ), f"Unexpected dims: {kinematic_array.dims}"

    @pytest.mark.parametrize(
        "valid_dataset_with_nan, expected_nans_per_individual",
        [
            (
                "valid_poses_dataset_with_nan",
                {
                    "displacement": [30, 0],
                    "velocity": [36, 0],
                    "acceleration": [40, 0],
                    "speed": [18, 0],
                    "path_length": [1, 0],  # Adjusted for NaN in id_0
                },
            ),
            (
                "valid_bboxes_dataset_with_nan",
                {
                    "displacement": [10, 0],
                    "velocity": [12, 0],
                    "acceleration": [14, 0],
                    "speed": [6, 0],
                    "path_length": [0, 0],
                },
            ),
        ],
    )
    @pytest.mark.parametrize(
        "kinematic_variable",
        ["displacement", "velocity", "acceleration", "speed", "path_length"],
    )
    def test_kinematics_with_dataset_with_nans(
        self,
        valid_dataset_with_nan,
        expected_nans_per_individual,
        kinematic_variable,
        helpers,  # Assuming this is needed; remove if unused
        request,
    ):
        """Test kinematic computations with datasets containing NaNs."""
        valid_dataset = request.getfixturevalue(valid_dataset_with_nan)
        position = valid_dataset.position
        kinematic_func = {
            "displacement": compute_displacement,
            "velocity": compute_velocity,
            "acceleration": compute_acceleration,
            "speed": compute_speed,
            "path_length": compute_path_length,
        }[kinematic_variable]
        kinematic_array = kinematic_func(position)
        expected_nans = expected_nans_per_individual[kinematic_variable]
        actual_nans = np.isnan(kinematic_array).sum().item()
        expected_total_nans = sum(expected_nans)
        assert actual_nans == expected_total_nans, (
            f"{kinematic_variable}: Expected {expected_total_nans} NaNs, "
            f"got {actual_nans}"
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
        ["displacement", "velocity", "acceleration", "speed", "path_length"],
    )
    def test_kinematics_with_invalid_dataset(
        self, invalid_dataset, expected_exception, kinematic_variable, request
    ):
        """Test kinematic computations with invalid datasets."""
        with expected_exception:
            position = request.getfixturevalue(invalid_dataset).position
            kinematic_func = {
                "displacement": compute_displacement,
                "velocity": compute_velocity,
                "acceleration": compute_acceleration,
                "speed": compute_speed,
                "path_length": compute_path_length,
            }[kinematic_variable]
            kinematic_func(position)


@pytest.mark.parametrize(
    "order, expected_exception",
    [
        (0, pytest.raises(ValueError)),
        (-1, pytest.raises(ValueError)),
        (1.0, pytest.raises(TypeError)),
        ("1", pytest.raises(TypeError)),
    ],
)
def test_time_derivative_with_invalid_order(order, expected_exception):
    """Test that an error is raised when the order is non-positive."""
    data = np.arange(10)
    with expected_exception:
        compute_time_derivative(data, order=order)
