import itertools
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
    "dim, pairs, expected_data",
    [
        (
            "individuals",
            ("ind1", "ind2"),
            np.array(
                [
                    [
                        [1.0, 0.0, np.sqrt(2)],
                        [1.0, np.sqrt(2), 0.0],
                        [0.0, 1.0, 1.0],
                    ],
                    [
                        [np.sqrt(13), 0.0, np.sqrt(8)],
                        [1.0, np.sqrt(8), 0.0],
                        [0.0, np.sqrt(13), 1.0],
                    ],
                ]
            ),
        ),
        (
            "keypoints",
            ("key1", "key2"),
            np.array(
                [
                    [
                        [np.sqrt(2), 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, np.sqrt(2), 1.0],
                    ],
                    [
                        [np.sqrt(8), 0.0, np.sqrt(13)],
                        [1.0, np.sqrt(13), 0.0],
                        [0.0, np.sqrt(8), 1.0],
                    ],
                ]
            ),
        ),
    ],
)
def test_cdist_with_known_values(
    dim, pairs, expected_data, pairwise_distances_dataset
):
    """Test the computation of pairwise distances with known values."""
    core_dim = "keypoints" if dim == "individuals" else "individuals"
    input_dataarray = pairwise_distances_dataset.position
    expected = xr.DataArray(
        expected_data,
        coords=[
            input_dataarray.time.values,
            getattr(input_dataarray, core_dim).values,
            getattr(input_dataarray, core_dim).values,
        ],
        dims=["time", pairs[0], pairs[1]],
    )
    a = input_dataarray.sel({dim: pairs[0]})
    b = input_dataarray.sel({dim: pairs[1]})
    result = kinematics.cdist(a, b, dim)
    xr.testing.assert_equal(
        result,
        expected,
    )


@pytest.mark.parametrize(
    "selection_fn",
    [
        lambda position: (
            position.sel(individuals="ind1"),
            position.sel(individuals="ind2"),
        ),  # individuals dim is scalar
        lambda position: (
            position.where(
                position.individuals == "ind1", drop=True
            ).squeeze(),
            position.where(
                position.individuals == "ind2", drop=True
            ).squeeze(),
        ),  # individuals dim is 1D
        lambda position: (
            position.sel(individuals="ind1", keypoints="key1"),
            position.sel(individuals="ind2", keypoints="key1"),
        ),  # both individuals and keypoints dims are scalar
        lambda position: (
            position.where(position.keypoints == "key1", drop=True).sel(
                individuals="ind1"
            ),
            position.where(position.keypoints == "key1", drop=True).sel(
                individuals="ind2"
            ),
        ),  # keypoints dim is 1D
        lambda position: (
            position.drop_sel(keypoints=position.keypoints.values[1:])
            .squeeze(drop=True)
            .sel(individuals="ind1"),
            position.drop_sel(keypoints=position.keypoints.values[1:])
            .squeeze(drop=True)
            .sel(individuals="ind2"),
        ),  # missing core dim
    ],
    ids=[
        "dim_has_ndim_0",
        "dim_has_ndim_1",
        "core_dim_has_ndim_0",
        "core_dim_has_ndim_1",
        "missing_core_dim",
    ],
)
def test_cdist_with_single_dim_inputs(
    pairwise_distances_dataset, selection_fn
):
    """Test that the computation of pairwise distances
    works regardless of whether the input DataArrays have
    ```dim``` and ```core_dim``` being either scalar (ndim=0)
    or 1D (ndim=1), or if ``core_dim`` is missing.
    """
    position = pairwise_distances_dataset.position
    a, b = selection_fn(position)
    with does_not_raise():
        kinematics.cdist(a, b, "individuals")


def expected_pairwise_distances(pairs, input_ds, dim):
    """Return a list of the expected data variable names
    for pairwise distances tests.
    """
    if pairs is None:
        paired_elements = list(
            itertools.combinations(getattr(input_ds, dim).values, 2)
        )
    else:
        paired_elements = [
            (elem1, elem2)
            for elem1, elem2_list in pairs.items()
            for elem2 in (
                [elem2_list] if isinstance(elem2_list, str) else elem2_list
            )
        ]
    expected_data = [
        f"dist_{elem1}_{elem2}" for elem1, elem2 in paired_elements
    ]
    return expected_data


@pytest.mark.parametrize(
    "dim, pairs",
    [
        ("individuals", {"ind1": ["ind2"]}),  # list input
        ("individuals", {"ind1": "ind2"}),  # string input
        ("individuals", {"ind1": ["ind2", "ind3"], "ind2": "ind3"}),
        ("individuals", None),  # all pairs
        ("keypoints", {"key1": ["key2"]}),  # list input
        ("keypoints", {"key1": "key2"}),  # string input
        ("keypoints", {"key1": ["key2", "key3"], "key2": "key3"}),
        ("keypoints", None),  # all pairs
    ],
)
def test_compute_pairwise_distances_with_valid_pairs(
    pairwise_distances_dataset, dim, pairs
):
    """Test that the expected pairwise distances are computed
    for valid ``pairs`` inputs.
    """
    result = getattr(kinematics, f"compute_inter{dim[:-1]}_distances")(
        pairwise_distances_dataset.position, pairs=pairs
    )
    expected_data_vars = expected_pairwise_distances(
        pairs, pairwise_distances_dataset, dim
    )
    if isinstance(result, dict):
        assert set(result.keys()) == set(expected_data_vars)
    else:  # expect single DataArray
        assert isinstance(result, xr.DataArray)


def test_compute_pairwise_distances_with_invalid_dim(
    pairwise_distances_dataset,
):
    """Test that an error is raised when an invalid dimension is passed."""
    with pytest.raises(ValueError):
        kinematics._compute_pairwise_distances(
            pairwise_distances_dataset.position, "invalid_dim"
        )
