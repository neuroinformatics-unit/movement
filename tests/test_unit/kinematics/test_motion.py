import numpy as np  # Added for NaN checking
import pytest
import xarray as xr

from movement.kinematics.motion import (
    compute_acceleration,
    compute_displacement,
    compute_speed,
    compute_velocity,
)


class TestComputeKinematics:
    """Test suite for computing kinematic variables."""

    @pytest.mark.parametrize(
        "valid_dataset", ["valid_poses_dataset", "valid_bboxes_dataset"]
    )
    @pytest.mark.parametrize(
        "kinematic_variable",
        ["displacement", "velocity", "acceleration", "speed"],
    )
    def test_kinematics(self, valid_dataset, kinematic_variable, request):
        """Test kinematic computations with valid datasets."""
        position = request.getfixturevalue(valid_dataset).position
        # Map kinematic_variable to the actual function
        kinematic_func = {
            "displacement": compute_displacement,
            "velocity": compute_velocity,
            "acceleration": compute_acceleration,
            "speed": compute_speed,
        }[kinematic_variable]
        kinematic_array = kinematic_func(position)
        # Add assertions as needed (e.g., check output type, dims)
        assert isinstance(kinematic_array, xr.DataArray)

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
                },
            ),
            (
                "valid_bboxes_dataset_with_nan",
                {
                    "displacement": [10, 0],
                    "velocity": [12, 0],
                    "acceleration": [14, 0],
                    "speed": [6, 0],
                },
            ),
        ],
    )
    @pytest.mark.parametrize(
        "kinematic_variable",
        ["displacement", "velocity", "acceleration", "speed"],
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
        }[kinematic_variable]
        kinematic_array = kinematic_func(position)
        expected_nans = expected_nans_per_individual[kinematic_variable]
        # Add NaN-checking logic
        actual_nans = (
            np.isnan(kinematic_array).sum().item()
        )  # Total NaNs in array
        expected_total_nans = sum(
            expected_nans
        )  # Sum of expected NaNs per individual
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
        ["displacement", "velocity", "acceleration", "speed"],
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
            }[kinematic_variable]
            kinematic_func(position)
