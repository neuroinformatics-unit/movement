import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement import kinematics


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
        (0, 10, pytest.raises(ValueError)),  # stop > n_frames
        (-1, 9, pytest.raises(ValueError)),  # start < 0
        (9, 0, pytest.raises(ValueError)),  # start > stop
        ("text", 9, pytest.raises(TypeError)),  # start is not a number
        (0, [0, 1], pytest.raises(TypeError)),  # stop is not a number
    ],
)
def test_compute_path_length_across_time_ranges(
    valid_poses_dataset_uniform_linear_motion,
    start,
    stop,
    expected_exception,
):
    """Test that the path length is computed correctly for a uniform linear
    motion case.
    """
    position = valid_poses_dataset_uniform_linear_motion.position
    with expected_exception:
        path_length = kinematics.compute_path_length(
            position, start=start, stop=stop, nan_policy="scale"
        )
        # Expected number of steps (displacements) in selected time range
        num_steps = 9  # full time range: 10 frames - 1
        if start is not None:
            num_steps -= np.ceil(start)
        if stop is not None:
            num_steps -= 9 - np.floor(stop)
        # Each step has a magnitude of sqrt(2) in x-y space
        expected_path_length = xr.DataArray(
            np.ones((2, 3)) * np.sqrt(2) * num_steps,
            dims=["individuals", "keypoints"],
            coords={
                "individuals": position.coords["individuals"],
                "keypoints": position.coords["keypoints"],
            },
        )
        xr.testing.assert_allclose(path_length, expected_path_length)


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


@pytest.mark.parametrize(
    "dim, expected_data",
    [
        (
            "individuals",
            np.array(
                [
                    [
                        [0.0, 1.0, 1.0],
                        [1.0, np.sqrt(2), 0.0],
                        [1.0, 2.0, np.sqrt(2)],
                    ],
                    [
                        [2.0, np.sqrt(5), 1.0],
                        [3.0, np.sqrt(10), 2.0],
                        [np.sqrt(5), np.sqrt(8), np.sqrt(2)],
                    ],
                ]
            ),
        ),
        (
            "keypoints",
            np.array(
                [[[1.0, 1.0], [1.0, 1.0]], [[1.0, np.sqrt(5)], [3.0, 1.0]]]
            ),
        ),
    ],
)
def test_cdist_with_known_values(
    dim, expected_data, valid_poses_dataset_uniform_linear_motion
):
    """Test the computation of pairwise distances with known values."""
    labels_dim = "keypoints" if dim == "individuals" else "individuals"
    input_dataarray = valid_poses_dataset_uniform_linear_motion.position.sel(
        time=slice(0, 1)
    )  # Use only the first two frames for simplicity
    pairs = input_dataarray[dim].values[:2]
    expected = xr.DataArray(
        expected_data,
        coords=[
            input_dataarray.time.values,
            getattr(input_dataarray, labels_dim).values,
            getattr(input_dataarray, labels_dim).values,
        ],
        dims=["time", pairs[0], pairs[1]],
    )
    a = input_dataarray.sel({dim: pairs[0]})
    b = input_dataarray.sel({dim: pairs[1]})
    result = kinematics._cdist(a, b, dim)
    xr.testing.assert_equal(
        result,
        expected,
    )


@pytest.mark.parametrize(
    "valid_dataset",
    [
        "valid_poses_dataset_uniform_linear_motion",
        "valid_bboxes_dataset",
    ],
)
@pytest.mark.parametrize(
    "selection_fn",
    [
        # individuals dim is scalar,
        # poses: multiple keypoints
        # bboxes: missing keypoints dim
        # e.g. comparing 2 individuals from the same data array
        lambda position: (
            position.isel(individuals=0),
            position.isel(individuals=1),
        ),
        # individuals dim is 1D
        # poses: multiple keypoints
        # bboxes: missing keypoints dim
        # e.g. comparing 2 single-individual data arrays
        lambda position: (
            position.where(
                position.individuals == position.individuals[0], drop=True
            ).squeeze(),
            position.where(
                position.individuals == position.individuals[1], drop=True
            ).squeeze(),
        ),
        # both individuals and keypoints dims are scalar (poses only)
        # e.g. comparing 2 individuals from the same data array,
        # at the same keypoint
        lambda position: (
            position.isel(individuals=0, keypoints=0),
            position.isel(individuals=1, keypoints=0),
        ),
        # individuals dim is scalar, keypoints dim is 1D (poses only)
        # e.g. comparing 2 single-individual, single-keypoint data arrays
        lambda position: (
            position.where(
                position.keypoints == position.keypoints[0], drop=True
            ).isel(individuals=0),
            position.where(
                position.keypoints == position.keypoints[0], drop=True
            ).isel(individuals=1),
        ),
    ],
    ids=[
        "dim_has_ndim_0",
        "dim_has_ndim_1",
        "labels_dim_has_ndim_0",
        "labels_dim_has_ndim_1",
    ],
)
def test_cdist_with_single_dim_inputs(valid_dataset, selection_fn, request):
    """Test that the pairwise distances data array is successfully
     returned regardless of whether the input DataArrays have
    ``dim`` ("individuals") and ``labels_dim`` ("keypoints")
    being either scalar (ndim=0) or 1D (ndim=1),
    or if ``labels_dim`` is missing.
    """
    if request.node.callspec.id not in [
        "labels_dim_has_ndim_0-valid_bboxes_dataset",
        "labels_dim_has_ndim_1-valid_bboxes_dataset",
    ]:  # Skip tests with keypoints dim for bboxes
        valid_dataset = request.getfixturevalue(valid_dataset)
        position = valid_dataset.position
        a, b = selection_fn(position)
        assert isinstance(kinematics._cdist(a, b, "individuals"), xr.DataArray)


@pytest.mark.parametrize(
    "dim, pairs, expected_data_vars",
    [
        ("individuals", {"id_1": ["id_2"]}, None),  # list input
        ("individuals", {"id_1": "id_2"}, None),  # string input
        (
            "individuals",
            {"id_1": ["id_2"], "id_2": "id_1"},
            [("id_1", "id_2"), ("id_2", "id_1")],
        ),
        ("individuals", "all", None),  # all pairs
        ("keypoints", {"centroid": ["left"]}, None),  # list input
        ("keypoints", {"centroid": "left"}, None),  # string input
        (
            "keypoints",
            {"centroid": ["left"], "left": "right"},
            [("centroid", "left"), ("left", "right")],
        ),
        (
            "keypoints",
            "all",
            [("centroid", "left"), ("centroid", "right"), ("left", "right")],
        ),  # all pairs
    ],
)
def test_compute_pairwise_distances_with_valid_pairs(
    valid_poses_dataset_uniform_linear_motion, dim, pairs, expected_data_vars
):
    """Test that the expected pairwise distances are computed
    for valid ``pairs`` inputs.
    """
    result = kinematics.compute_pairwise_distances(
        valid_poses_dataset_uniform_linear_motion.position, dim, pairs
    )
    if isinstance(result, dict):
        expected_data_vars = [
            f"dist_{pair[0]}_{pair[1]}" for pair in expected_data_vars
        ]
        assert set(result.keys()) == set(expected_data_vars)
    else:  # expect single DataArray
        assert isinstance(result, xr.DataArray)


@pytest.mark.parametrize(
    "ds, dim, pairs",
    [
        (
            "valid_poses_dataset_uniform_linear_motion",
            "invalid_dim",
            {"id_1": "id_2"},
        ),  # invalid dim
        (
            "valid_poses_dataset_uniform_linear_motion",
            "keypoints",
            "invalid_string",
        ),  # invalid pairs
        (
            "valid_poses_dataset_uniform_linear_motion",
            "individuals",
            {},
        ),  # empty pairs
        ("missing_dim_poses_dataset", "keypoints", "all"),  # invalid dataset
        (
            "missing_dim_bboxes_dataset",
            "individuals",
            "all",
        ),  # invalid dataset
    ],
)
def test_compute_pairwise_distances_with_invalid_input(
    ds, dim, pairs, request
):
    """Test that an error is raised for invalid inputs."""
    with pytest.raises(ValueError):
        kinematics.compute_pairwise_distances(
            request.getfixturevalue(ds).position, dim, pairs
        )
