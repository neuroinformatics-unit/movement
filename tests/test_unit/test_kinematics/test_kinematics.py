from itertools import product

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement import kinematics
from movement.kinematics import (
    compute_acceleration,
    compute_directional_change,
    compute_path_emax,
    compute_path_sinuosity,
    compute_speed,
    compute_time_derivative,
    compute_turning_angle,
    compute_velocity,
)


@pytest.mark.parametrize(
    "kinematic_variable",
    [
        "forward_displacement",
        "backward_displacement",
        "velocity",
        "acceleration",
        "speed",
    ],
)
class TestComputeKinematics:
    """Test ``compute_[kinematic_variable]`` with valid and invalid inputs."""

    forward_displacement = [
        np.vstack([np.ones((9, 2)), np.zeros((1, 2))]),
        np.multiply(
            np.vstack([np.ones((9, 2)), np.zeros((1, 2))]),
            np.array([1, -1]),
        ),
    ]  # [Individual 0, Individual 1]
    expected_kinematics = {
        "forward_displacement": forward_displacement,
        "backward_displacement": [
            np.roll(-forward_displacement[0], 1, axis=0),
            np.roll(-forward_displacement[1], 1, axis=0),
        ],
        "velocity": [
            np.ones((10, 2)),
            np.multiply(np.ones((10, 2)), np.array([1, -1])),
        ],
        "acceleration": [np.zeros((10, 2)), np.zeros((10, 2))],
        "speed": [np.ones(10) * np.sqrt(2), np.ones(10) * np.sqrt(2)],
    }  # 2D: n_frames, n_space_dims

    @pytest.mark.parametrize(
        "valid_dataset", ["valid_poses_dataset", "valid_bboxes_dataset"]
    )
    def test_kinematics(self, valid_dataset, kinematic_variable, request):
        """Test computed kinematics for a uniform linear motion case.
        See the ``valid_poses_dataset`` and ``valid_bboxes_dataset`` fixtures
        for details.
        """
        # Compute kinematic array from input dataset
        position = request.getfixturevalue(valid_dataset).position
        kinematic_array = getattr(kinematics, f"compute_{kinematic_variable}")(
            position
        )
        assert kinematic_array.name == kinematic_variable
        # Figure out which dimensions to expect in kinematic_array
        # and in the final xarray.DataArray
        expected_dims = ["time", "individual"]
        if kinematic_variable in [
            "forward_displacement",
            "backward_displacement",
            "velocity",
            "acceleration",
        ]:
            expected_dims.insert(1, "space")
        # Build expected data array from the expected numpy array
        expected_array = xr.DataArray(
            # Stack along the "individual" axis
            np.stack(
                self.expected_kinematics.get(kinematic_variable), axis=-1
            ),
            dims=expected_dims,
        )
        if "keypoint" in position.coords:
            expected_array = expected_array.expand_dims(
                {"keypoint": position.coords["keypoint"].size}
            )
            expected_dims.insert(-1, "keypoint")
            expected_array = expected_array.transpose(*expected_dims)
        # Compare the values of the kinematic_array against the expected_array
        np.testing.assert_allclose(
            kinematic_array.values, expected_array.values
        )

    @pytest.mark.filterwarnings(
        "ignore:The result may be unreliable.*:UserWarning"
    )
    @pytest.mark.parametrize(
        "valid_dataset_with_nan, expected_nans_per_individual",
        [
            (
                "valid_poses_dataset_with_nan",
                {
                    "forward_displacement": [30, 0],
                    "backward_displacement": [30, 0],
                    "velocity": [36, 0],
                    "acceleration": [40, 0],
                    "speed": [18, 0],
                },
            ),
            (
                "valid_bboxes_dataset_with_nan",
                {
                    "forward_displacement": [10, 0],
                    "backward_displacement": [10, 0],
                    "velocity": [12, 0],
                    "acceleration": [14, 0],
                    "speed": [6, 0],
                },
            ),
        ],
    )
    def test_kinematics_with_dataset_with_nans(
        self,
        valid_dataset_with_nan,
        expected_nans_per_individual,
        kinematic_variable,
        helpers,
        request,
    ):
        """Test kinematics computation for a dataset with nans.
        See the ``valid_poses_dataset_with_nan`` and
        ``valid_bboxes_dataset_with_nan`` fixtures for details.
        """
        # compute kinematic array
        valid_dataset = request.getfixturevalue(valid_dataset_with_nan)
        position = valid_dataset.position
        kinematic_array = getattr(kinematics, f"compute_{kinematic_variable}")(
            position
        )
        assert kinematic_array.name == kinematic_variable
        # compute n nans in kinematic array per individual
        n_nans_kinematics_per_indiv = [
            helpers.count_nans(kinematic_array.isel(individual=i))
            for i in range(valid_dataset.sizes["individual"])
        ]
        # assert n nans in kinematic array per individual matches expected
        assert (
            n_nans_kinematics_per_indiv
            == expected_nans_per_individual[kinematic_variable]
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
    def test_kinematics_with_invalid_dataset(
        self, invalid_dataset, expected_exception, kinematic_variable, request
    ):
        """Test kinematics computation with an invalid dataset."""
        with expected_exception:
            position = request.getfixturevalue(invalid_dataset).position
            getattr(kinematics, f"compute_{kinematic_variable}")(position)


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
        kinematics.compute_time_derivative(data, order=order)


def test_forward_displacement_with_multiindex_coords():
    """Test compute_forward_displacement with DataArray from pandas MultiIndex.

    This is a test for the fix of issue
    [#794](https://github.com/neuroinformatics-unit/movement/issues/794).
    When creating an xarray DataArray from a MultiIndex pandas DataFrame
    using `.to_xarray()`, the resulting coordinates retain the
    `_no_setting_name` flag from pandas MultiIndex levels.
    This caused a RuntimeError when xarray's `.reindex()` tried to set
    `.name` on these indices.
    """
    # Create DataFrame with MultiIndex (reproduction from issue #794)
    frames = range(3)
    space = ["x", "y"]
    keypoints = ["centroid"]
    individuals = ["bird001"]

    df = pd.DataFrame(
        list(product(frames, space, keypoints, individuals)),
        columns=["time", "space", "keypoint", "individual"],
    )
    df["position"] = np.random.rand(len(df))

    # Convert to xarray DataArray via pandas MultiIndex
    # This retains the _no_setting_name flag that caused the bug
    position = (
        df.loc[:, ["time", "space", "keypoint", "individual", "position"]]
        .set_index(["time", "space", "keypoint", "individual"])["position"]
        .to_xarray()
    )

    # This should not raise RuntimeError
    result = kinematics.compute_forward_displacement(position)

    # Verify correct output
    assert result.name == "forward_displacement"
    assert result.dims == position.dims
    assert result.shape == position.shape


# List of tuples containing (function, extra_arguments_dict)
FUNCTIONS_TO_TEST = [
    (compute_velocity, {}),
    (compute_acceleration, {}),
    (compute_time_derivative, {"order": 1}),
    (compute_turning_angle, {}),
    (compute_directional_change, {}),
    (compute_path_emax, {}),
    (compute_path_sinuosity, {}),
    (compute_speed, {}),
]


@pytest.mark.parametrize("func, kwargs", FUNCTIONS_TO_TEST)
def test_kinematics_preserves_attributes(func, kwargs):
    # Create mock data with a minimum of 3 time points
    data = xr.DataArray(
        np.random.rand(10, 2),
        dims=["time", "space"],
        coords={"time": np.arange(10), "space": ["x", "y"]},
    )
    # Attaching the attributes to track
    data.attrs = {"log": "test_log", "foo": 1, "units": "cm"}

    # Run the function with the mock data and any extra arguments
    result = func(data, **kwargs)

    # Verify the original non-unit attributes survived
    assert result.attrs.get("log") == "test_log"
    assert result.attrs.get("foo") == 1

    # Verify the original non-unit attributes survived
    assert result.attrs.get("log") == "test_log"
    assert result.attrs.get("foo") == 1

    # Check that "radians" didn't leak into functions where it shouldn't be
    if "units" in result.attrs and func.__name__ != "compute_turning_angle":
        assert result.attrs["units"] != "radians"
