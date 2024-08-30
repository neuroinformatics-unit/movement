import numpy as np
import pytest
import xarray as xr

from movement.analysis import kinematics


@pytest.mark.parametrize(
    "valid_dataset_uniform_linear_motion",
    [
        "valid_poses_dataset_uniform_linear_motion",
        "valid_bboxes_dataset",
    ],
)
@pytest.mark.parametrize(
    "kinematic_variable, expected_kinematics",
    [
        (
            "displacement",
            [
                np.vstack([np.zeros((1, 2)), np.ones((9, 2))]),  # Individual 0
                np.multiply(
                    np.vstack([np.zeros((1, 2)), np.ones((9, 2))]),
                    np.array([1, -1]),
                ),  # Individual 1
            ],
        ),
        (
            "velocity",
            [
                np.ones((10, 2)),  # Individual 0
                np.multiply(
                    np.ones((10, 2)), np.array([1, -1])
                ),  # Individual 1
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
def test_kinematics_uniform_linear_motion(
    valid_dataset_uniform_linear_motion,
    kinematic_variable,
    expected_kinematics,  # 2D: n_frames, n_space_dims
    request,
):
    """Test computed kinematics for a uniform linear motion case.

    Uniform linear motion means the individuals move along a line
    at constant velocity.

    We consider 2 individuals ("id_0" and "id_1"),
    tracked for 10 frames, along x and y:
    - id_0 moves along x=y line from the origin
    - id_1 moves along x=-y line from the origin
    - they both move one unit (pixel) along each axis in each frame

    If the dataset is a poses dataset, we consider 3 keypoints per individual
    (centroid, left, right), that are always in front of the centroid keypoint
    at 45deg from the trajectory.
    """
    # Compute kinematic array from input dataset
    position = request.getfixturevalue(
        valid_dataset_uniform_linear_motion
    ).position
    kinematic_array = getattr(kinematics, f"compute_{kinematic_variable}")(
        position
    )

    # Build expected data array from the expected numpy array
    expected_array = xr.DataArray(
        np.stack(expected_kinematics, axis=1),
        # Stack along the "individuals" axis
        dims=["time", "individuals", "space"],
    )
    if "keypoints" in position.coords:
        expected_array = expected_array.expand_dims(
            {"keypoints": position.coords["keypoints"].size}
        )
        expected_array = expected_array.transpose(
            "time", "individuals", "keypoints", "space"
        )

    # Compare the values of the kinematic_array against the expected_array
    np.testing.assert_allclose(kinematic_array.values, expected_array.values)


@pytest.mark.parametrize(
    "valid_dataset_with_nan",
    [
        "valid_poses_dataset_with_nan",
        "valid_bboxes_dataset_with_nan",
    ],
)
@pytest.mark.parametrize(
    "kinematic_variable, expected_nans_per_individual",
    [
        ("displacement", [5, 0]),  # individual 0, individual 1
        ("velocity", [6, 0]),
        ("acceleration", [7, 0]),
    ],
)
def test_kinematics_with_dataset_with_nans(
    valid_dataset_with_nan,
    kinematic_variable,
    expected_nans_per_individual,
    helpers,
    request,
):
    """Test kinematics computation for a dataset with nans.

    We test that the kinematics can be computed and that the number
    of nan values in the kinematic array is as expected.

    """
    # compute kinematic array
    valid_dataset = request.getfixturevalue(valid_dataset_with_nan)
    position = valid_dataset.position
    kinematic_array = getattr(kinematics, f"compute_{kinematic_variable}")(
        position
    )

    # compute n nans in kinematic array per individual
    n_nans_kinematics_per_indiv = [
        helpers.count_nans(kinematic_array.isel(individuals=i))
        for i in range(valid_dataset.sizes["individuals"])
    ]

    # expected nans per individual adjusted for space and keypoints dimensions
    expected_nans_adjusted = [
        n
        * valid_dataset.sizes["space"]
        * valid_dataset.sizes.get("keypoints", 1)
        for n in expected_nans_per_individual
    ]
    # check number of nans per individual is as expected in kinematic array
    np.testing.assert_array_equal(
        n_nans_kinematics_per_indiv, expected_nans_adjusted
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
    """Test kinematics computation with an invalid dataset."""
    with expected_exception:
        position = request.getfixturevalue(invalid_dataset).position
        getattr(kinematics, f"compute_{kinematic_variable}")(position)


@pytest.mark.parametrize("order", [0, -1, 1.0, "1"])
def test_approximate_derivative_with_invalid_order(order):
    """Test that an error is raised when the order is non-positive."""
    data = np.arange(10)
    expected_exception = ValueError if isinstance(order, int) else TypeError
    with pytest.raises(expected_exception):
        kinematics.compute_time_derivative(data, order=order)


class TestNavigation:
    """Test suite for navigation-related functions in the kinematics module."""

    @pytest.fixture
    def mock_data_array(self):
        """Return a mock DataArray containing four known head orientations."""
        time = [0, 1, 2, 3]
        individuals = ["individual_0"]
        keypoints = ["left_ear", "right_ear", "nose"]
        space = ["x", "y"]

        ds = xr.DataArray(
            [
                [[[1, 0], [-1, 0], [0, -1]]],  # time 0
                [[[0, 1], [0, -1], [1, 0]]],  # time 1
                [[[-1, 0], [1, 0], [0, 1]]],  # time 2
                [[[0, -1], [0, 1], [-1, 0]]],  # time 3
            ],
            dims=["time", "individuals", "keypoints", "space"],
            coords={
                "time": time,
                "individuals": individuals,
                "keypoints": keypoints,
                "space": space,
            },
        )
        return ds

    @pytest.fixture
    def mock_data_array_3D(self):
        """Return a 3D DataArray containing a known head orientation."""
        time = [0]
        individuals = ["individual_0"]
        keypoints = ["left", "right"]
        space = ["x", "y", "z"]

        ds = xr.DataArray(
            [
                [[[1, 0, 0], [-1, 0, 0]]],
            ],
            dims=["time", "individuals", "keypoints", "space"],
            coords={
                "time": time,
                "individuals": individuals,
                "keypoints": keypoints,
                "space": space,
            },
        )
        return ds

    def test_compute_head_direction_vector(
        self, mock_data_array, mock_data_array_3D
    ):
        """Test that the correct head direction vectors
        are computed from a basic mock dataset.
        """
        # Test that validators work

        # Catch incorrect datatype
        with pytest.raises(TypeError, match="must be an xarray.DataArray"):
            kinematics.compute_head_direction_vector(
                mock_data_array.values, "left_ear", "right_ear"
            )

        # Catch incorrect dimensions
        with pytest.raises(
            AttributeError, match="'time', 'space', and 'keypoints'"
        ):
            mock_data_keypoint = mock_data_array.sel(
                keypoints="nose", drop=True
            )
            kinematics.compute_head_direction_vector(
                mock_data_keypoint, "left_ear", "right_ear"
            )

        # Catch identical left and right keypoints
        with pytest.raises(ValueError, match="keypoints may not be identical"):
            kinematics.compute_head_direction_vector(
                mock_data_array, "left_ear", "left_ear"
            )

        # Catch incorrect spatial dimensions
        with pytest.raises(
            ValueError, match="must have 2 (and only 2) spatial dimensions"
        ):
            kinematics.compute_head_direction_vector(
                mock_data_array_3D, "left", "right"
            )

        # Test that output contains correct datatype, dimensions, and values
        head_vector = kinematics.compute_head_direction_vector(
            mock_data_array, "left_ear", "right_ear"
        )
        known_vectors = np.array([[[0, 2]], [[-2, 0]], [[0, -2]], [[2, 0]]])

        assert (
            isinstance(head_vector, xr.DataArray)
            and ("space" in head_vector.dims)
            and ("keypoints" not in head_vector.dims)
        )
        assert np.equal(head_vector.values, known_vectors).all()