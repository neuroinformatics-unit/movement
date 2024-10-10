import re
from contextlib import nullcontext as does_not_raise

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
        (
            "speed",  # magnitude of velocity
            [
                np.ones(10) * np.sqrt(2),  # Individual 0
                np.ones(10) * np.sqrt(2),  # Individual 1
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

    # Figure out which dimensions to expect in kinematic_array
    # and in the final xarray.DataArray
    expected_dims = ["time", "individuals"]
    if kinematic_variable in ["displacement", "velocity", "acceleration"]:
        expected_dims.append("space")

    # Build expected data array from the expected numpy array
    expected_array = xr.DataArray(
        # Stack along the "individuals" axis
        np.stack(expected_kinematics, axis=1),
        dims=expected_dims,
    )
    if "keypoints" in position.coords:
        expected_array = expected_array.expand_dims(
            {"keypoints": position.coords["keypoints"].size}
        )
        expected_dims.insert(2, "keypoints")
        expected_array = expected_array.transpose(*expected_dims)

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
        ("speed", [6, 0]),
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
    if "space" in kinematic_array.dims:
        n_space_dims = position.sizes["space"]
    else:
        n_space_dims = 1

    expected_nans_adjusted = [
        n * n_space_dims * valid_dataset.sizes.get("keypoints", 1)
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
        "speed",
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


@pytest.mark.parametrize(
    "start, stop, expected_exception",
    [
        # full time ranges
        (None, None, does_not_raise()),
        (0, None, does_not_raise()),
        (None, 9, does_not_raise()),
        (0, 9, does_not_raise()),
        # partial time ranges
        (1, 8, does_not_raise()),
        (1.5, 8.5, does_not_raise()),
        (2, None, does_not_raise()),
        (None, 6.3, does_not_raise()),
        # invalid time ranges
        (
            0,
            10,
            pytest.raises(
                ValueError, match="stop time 10 is outside the time range"
            ),
        ),
        (
            -1,
            9,
            pytest.raises(
                ValueError, match="start time -1 is outside the time range"
            ),
        ),
        (
            9,
            0,
            pytest.raises(
                ValueError,
                match="start time must be earlier than the stop time",
            ),
        ),
        (
            "text",
            9,
            pytest.raises(
                TypeError, match="Expected a numeric value for start"
            ),
        ),
        (
            0,
            [0, 1],
            pytest.raises(
                TypeError, match="Expected a numeric value for stop"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "nan_policy",
    ["drop", "scale"],  # results should be same for both here
)
def test_path_length_across_time_ranges(
    valid_poses_dataset_uniform_linear_motion,
    start,
    stop,
    nan_policy,
    expected_exception,
):
    """Test path length computation for a uniform linear motion case,
    across different time ranges.

    The test dataset ``valid_poses_dataset_uniform_linear_motion``
    contains 2 individuals ("id_0" and "id_1"), moving
    along x=y and x=-y lines, respectively, at a constant velocity.
    At each frame they cover a distance of sqrt(2) in x-y space, so in total
    we expect a path length of sqrt(2) * num_segments, where num_segments is
    the number of selected frames minus 1.
    """
    position = valid_poses_dataset_uniform_linear_motion.position
    with expected_exception:
        path_length = kinematics.compute_path_length(
            position, start=start, stop=stop, nan_policy=nan_policy
        )

        # Expected number of segments (displacements) in selected time range
        num_segments = 9  # full time range: 10 frames - 1
        if start is not None:
            num_segments -= np.ceil(start)
        if stop is not None:
            num_segments -= 9 - np.floor(stop)
        print("num_segments", num_segments)

        expected_path_length = xr.DataArray(
            np.ones((2, 3)) * np.sqrt(2) * num_segments,
            dims=["individuals", "keypoints"],
            coords={
                "individuals": position.coords["individuals"],
                "keypoints": position.coords["keypoints"],
            },
        )
        xr.testing.assert_allclose(path_length, expected_path_length)


@pytest.mark.parametrize(
    "nan_policy, expected_path_lengths_id_1, expected_exception",
    [
        (
            "drop",
            {
                # 9 segments - 1 missing on edge
                "centroid": np.sqrt(2) * 8,
                # missing mid frames should have no effect
                "left": np.sqrt(2) * 9,
                "right": np.nan,  # all frames missing
            },
            does_not_raise(),
        ),
        (
            "scale",
            {
                # scaling should restore "true" path length
                "centroid": np.sqrt(2) * 9,
                "left": np.sqrt(2) * 9,
                "right": np.nan,  # all frames missing
            },
            does_not_raise(),
        ),
        (
            "invalid",  # invalid value for nan_policy
            {},
            pytest.raises(ValueError, match="Invalid value for nan_policy"),
        ),
    ],
)
def test_path_length_with_nans(
    valid_poses_dataset_uniform_linear_motion_with_nans,
    nan_policy,
    expected_path_lengths_id_1,
    expected_exception,
):
    """Test path length computation for a uniform linear motion case,
    with varying number of missing values per individual and keypoint.

    The test dataset ``valid_poses_dataset_uniform_linear_motion_with_nans``
    contains 2 individuals ("id_0" and "id_1"), moving
    along x=y and x=-y lines, respectively, at a constant velocity.
    At each frame they cover a distance of sqrt(2) in x-y space.

    Individual "id_1" has some missing values per keypoint:
    - "centroid" is missing a value on the very first frame
    - "left" is missing 5 values in middle frames (not at the edges)
    - "right" is missing values in all frames

    Individual "id_0" has no missing values.

    Because the underlying motion is uniform linear, the "scale" policy should
    perfectly restore the path length for individual "id_1" to its true value.
    The "drop" policy should do likewise if frames are missing in the middle,
    but will not count any missing frames at the edges.
    """
    position = valid_poses_dataset_uniform_linear_motion_with_nans.position
    with expected_exception:
        path_length = kinematics.compute_path_length(
            position,
            nan_policy=nan_policy,
        )
        # Initialise with expected path lengths for scenario without NaNs
        expected_array = xr.DataArray(
            np.ones((2, 3)) * np.sqrt(2) * 9,
            dims=["individuals", "keypoints"],
            coords={
                "individuals": position.coords["individuals"],
                "keypoints": position.coords["keypoints"],
            },
        )
        # insert expected path lengths for individual id_1
        for kpt, value in expected_path_lengths_id_1.items():
            target_loc = {"individuals": "id_1", "keypoints": kpt}
            expected_array.loc[target_loc] = value
        xr.testing.assert_allclose(path_length, expected_array)


@pytest.mark.parametrize(
    "nan_warn_threshold, expected_exception",
    [
        (1, does_not_raise()),
        (0.2, does_not_raise()),
        (-1, pytest.raises(ValueError, match="a number between 0 and 1")),
    ],
)
def test_path_length_warns_about_nans(
    valid_poses_dataset_uniform_linear_motion_with_nans,
    nan_warn_threshold,
    expected_exception,
    caplog,
):
    """Test that a warning is raised when the number of missing values
    exceeds a given threshold.

    See the docstring of ``test_path_length_with_nans`` for details
    about what's in the dataset.
    """
    position = valid_poses_dataset_uniform_linear_motion_with_nans.position
    with expected_exception:
        kinematics.compute_path_length(
            position, nan_warn_threshold=nan_warn_threshold
        )

        if (nan_warn_threshold > 0.1) and (nan_warn_threshold < 0.5):
            # Make sure that a warning was emitted
            assert caplog.records[0].levelname == "WARNING"
            assert "The result may be unreliable" in caplog.records[0].message
            # Make sure that the NaN report only mentions
            # the individual and keypoint that violate the threshold
            assert caplog.records[1].levelname == "INFO"
            assert "Individual: id_1" in caplog.records[1].message
            assert "Individual: id_2" not in caplog.records[1].message
            assert "left: 5/10 (50.0%)" in caplog.records[1].message
            assert "right: 10/10 (100.0%)" in caplog.records[1].message
            assert "centroid" not in caplog.records[1].message


@pytest.fixture
def valid_data_array_for_forward_vector():
    """Return a position data array for an individual with 3 keypoints
    (left ear, right ear and nose), tracked for 4 frames, in x-y space.
    """
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
def invalid_input_type_for_forward_vector(valid_data_array_for_forward_vector):
    """Return a numpy array of position values by individual, per keypoint,
    over time.
    """
    return valid_data_array_for_forward_vector.values


@pytest.fixture
def invalid_dimensions_for_forward_vector(valid_data_array_for_forward_vector):
    """Return a position DataArray in which the ``keypoints`` dimension has
    been dropped.
    """
    return valid_data_array_for_forward_vector.sel(keypoints="nose", drop=True)


@pytest.fixture
def invalid_spatial_dimensions_for_forward_vector(
    valid_data_array_for_forward_vector,
):
    """Return a position DataArray containing three spatial dimensions."""
    dataarray_3d = valid_data_array_for_forward_vector.pad(
        space=(0, 1), constant_values=0
    )
    return dataarray_3d.assign_coords(space=["x", "y", "z"])


@pytest.fixture
def valid_data_array_for_forward_vector_with_nans(
    valid_data_array_for_forward_vector,
):
    """Return a position DataArray where position values are NaN for the
    ``left_ear`` keypoint at time ``1``.
    """
    nan_dataarray = valid_data_array_for_forward_vector.where(
        (valid_data_array_for_forward_vector.time != 1)
        | (valid_data_array_for_forward_vector.keypoints != "left_ear")
    )
    return nan_dataarray


def test_compute_forward_vector(valid_data_array_for_forward_vector):
    """Test that the correct output forward direction vectors
    are computed from a valid mock dataset.
    """
    forward_vector = kinematics.compute_forward_vector(
        valid_data_array_for_forward_vector,
        "left_ear",
        "right_ear",
        camera_view="bottom_up",
    )
    forward_vector_flipped = kinematics.compute_forward_vector(
        valid_data_array_for_forward_vector,
        "left_ear",
        "right_ear",
        camera_view="top_down",
    )
    head_vector = kinematics.compute_head_direction_vector(
        valid_data_array_for_forward_vector,
        "left_ear",
        "right_ear",
        camera_view="bottom_up",
    )
    known_vectors = np.array([[[0, -1]], [[1, 0]], [[0, 1]], [[-1, 0]]])

    assert (
        isinstance(forward_vector, xr.DataArray)
        and ("space" in forward_vector.dims)
        and ("keypoints" not in forward_vector.dims)
    )
    assert np.equal(forward_vector.values, known_vectors).all()
    assert np.equal(forward_vector_flipped.values, known_vectors * -1).all()
    assert head_vector.equals(forward_vector)


@pytest.mark.parametrize(
    "input_data, expected_error, expected_match_str, keypoints",
    [
        (
            "invalid_input_type_for_forward_vector",
            TypeError,
            "must be an xarray.DataArray",
            ["left_ear", "right_ear"],
        ),
        (
            "invalid_dimensions_for_forward_vector",
            ValueError,
            "Input data must contain ['keypoints']",
            ["left_ear", "right_ear"],
        ),
        (
            "invalid_spatial_dimensions_for_forward_vector",
            ValueError,
            "must have exactly 2 spatial dimensions",
            ["left_ear", "right_ear"],
        ),
        (
            "valid_data_array_for_forward_vector",
            ValueError,
            "keypoints may not be identical",
            ["left_ear", "left_ear"],
        ),
    ],
)
def test_compute_forward_vector_with_invalid_input(
    input_data, keypoints, expected_error, expected_match_str, request
):
    """Test that ``compute_forward_vector`` catches errors
    correctly when passed invalid inputs.
    """
    # Get fixture
    input_data = request.getfixturevalue(input_data)

    # Catch error
    with pytest.raises(expected_error, match=re.escape(expected_match_str)):
        kinematics.compute_forward_vector(
            input_data, keypoints[0], keypoints[1]
        )


def test_nan_behavior_forward_vector(
    valid_data_array_for_forward_vector_with_nans,
):
    """Test that ``compute_forward_vector()`` generates the
    expected output for a valid input DataArray containing ``NaN``
    position values at a single time (``1``) and keypoint
    (``left_ear``).
    """
    forward_vector = kinematics.compute_forward_vector(
        valid_data_array_for_forward_vector_with_nans, "left_ear", "right_ear"
    )
    assert (
        np.isnan(forward_vector.values[1, 0, :]).all()
        and not np.isnan(forward_vector.values[[0, 2, 3], 0, :]).any()
    )
