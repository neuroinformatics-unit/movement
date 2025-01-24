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
        expected_dims.insert(1, "space")

    # Build expected data array from the expected numpy array
    expected_array = xr.DataArray(
        # Stack along the "individuals" axis
        np.stack(expected_kinematics, axis=-1),
        dims=expected_dims,
    )
    if "keypoints" in position.coords:
        expected_array = expected_array.expand_dims(
            {"keypoints": position.coords["keypoints"].size}
        )
        expected_dims.insert(-1, "keypoints")
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
    n_space_dims = (
        position.sizes["space"] if "space" in kinematic_array.dims else 1
    )
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


time_points_value_error = pytest.raises(
    ValueError,
    match="At least 2 time points are required to compute path length",
)


@pytest.mark.parametrize(
    "start, stop, expected_exception",
    [
        # full time ranges
        (None, None, does_not_raise()),
        (0, None, does_not_raise()),
        (0, 9, does_not_raise()),
        (0, 10, does_not_raise()),  # xarray.sel will truncate to 0, 9
        (-1, 9, does_not_raise()),  # xarray.sel will truncate to 0, 9
        # partial time ranges
        (1, 8, does_not_raise()),
        (1.5, 8.5, does_not_raise()),
        (2, None, does_not_raise()),
        # Empty time ranges
        (9, 0, time_points_value_error),  # start > stop
        ("text", 9, time_points_value_error),  # invalid start type
        # Time range too short
        (0, 0.5, time_points_value_error),
    ],
)
def test_path_length_across_time_ranges(
    valid_poses_dataset_uniform_linear_motion,
    start,
    stop,
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
            position, start=start, stop=stop
        )

        # Expected number of segments (displacements) in selected time range
        num_segments = 9  # full time range: 10 frames - 1
        start = max(0, start) if start is not None else 0
        stop = min(9, stop) if stop is not None else 9
        if start is not None:
            num_segments -= np.ceil(max(0, start))
        if stop is not None:
            stop = min(9, stop)
            num_segments -= 9 - np.floor(min(9, stop))

        expected_path_length = xr.DataArray(
            np.ones((3, 2)) * np.sqrt(2) * num_segments,
            dims=["keypoints", "individuals"],
            coords={
                "keypoints": position.coords["keypoints"],
                "individuals": position.coords["individuals"],
            },
        )
        xr.testing.assert_allclose(path_length, expected_path_length)


@pytest.mark.parametrize(
    "nan_policy, expected_path_lengths_id_1, expected_exception",
    [
        (
            "ffill",
            np.array([np.sqrt(2) * 8, np.sqrt(2) * 9, np.nan]),
            does_not_raise(),
        ),
        (
            "scale",
            np.array([np.sqrt(2) * 9, np.sqrt(2) * 9, np.nan]),
            does_not_raise(),
        ),
        (
            "invalid",  # invalid value for nan_policy
            np.zeros(3),
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
    The "ffill" policy should do likewise if frames are missing in the middle,
    but will not "correct" for missing values at the edges.
    """
    position = valid_poses_dataset_uniform_linear_motion_with_nans.position
    with expected_exception:
        path_length = kinematics.compute_path_length(
            position,
            nan_policy=nan_policy,
        )
        # Get path_length for individual "id_1" as a numpy array
        path_length_id_1 = path_length.sel(individuals="id_1").values
        # Check them against the expected values
        np.testing.assert_allclose(
            path_length_id_1, expected_path_lengths_id_1
        )


@pytest.mark.parametrize(
    "nan_warn_threshold, expected_exception",
    [
        (1, does_not_raise()),
        (0.2, does_not_raise()),
        (-1, pytest.raises(ValueError, match="between 0 and 1")),
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

    for output_array in [forward_vector, forward_vector_flipped, head_vector]:
        assert isinstance(output_array, xr.DataArray)
        for preserved_coord in ["time", "space", "individuals"]:
            assert np.all(
                output_array[preserved_coord]
                == valid_data_array_for_forward_vector[preserved_coord]
            )
        assert set(output_array["space"].values) == {"x", "y"}
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
    valid_data_array_for_forward_vector_with_nans: xr.DataArray,
):
    """Test that ``compute_forward_vector()`` generates the
    expected output for a valid input DataArray containing ``NaN``
    position values at a single time (``1``) and keypoint
    (``left_ear``).
    """
    nan_time = 1
    forward_vector = kinematics.compute_forward_vector(
        valid_data_array_for_forward_vector_with_nans, "left_ear", "right_ear"
    )
    # Check coord preservation
    for preserved_coord in ["time", "space", "individuals"]:
        assert np.all(
            forward_vector[preserved_coord]
            == valid_data_array_for_forward_vector_with_nans[preserved_coord]
        )
    assert set(forward_vector["space"].values) == {"x", "y"}
    # Should have NaN values in the forward vector at time 1 and left_ear
    nan_values = forward_vector.sel(time=nan_time)
    assert nan_values.shape == (1, 2)
    assert np.isnan(
        nan_values
    ).all(), "NaN values not returned where expected!"
    # Should have no NaN values in the forward vector in other positions
    assert not np.isnan(
        forward_vector.sel(
            time=[
                t
                for t in valid_data_array_for_forward_vector_with_nans.time
                if t != nan_time
            ]
        )
    ).any()


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


class TestForwardVectorAngle:
    """Test the compute_forward_vector_angle function.

    def compute_forward_vector_angle(
        data: xr.DataArray,
        left_keypoint: str,
        right_keypoint: str,
        reference_vector: xr.DataArray | np.ndarray | list | tuple = (1, 0),
        camera_view: Literal["top_down", "bottom_up"] = "top_down",
        in_radians: bool = False,
        angle_rotates: Literal[
            "ref to forward", "forward to ref"
        ] = "ref to forward",
    ) -> xr.DataArray:
    """

    x_axis = np.array([1.0, 0.0])
    y_axis = np.array([0.0, 1.0])
    sqrt_2 = np.sqrt(2.0)

    @staticmethod
    def push_into_range(
        numeric_values: xr.DataArray | np.ndarray,
        lower: float = -180.0,
        upper: float = 180.0,
    ) -> xr.DataArray | np.ndarray:
        """Translate values into the range (lower, upper].

        The interval width is the value ``upper - lower``.
        Each ``v`` in ``values`` that starts less than or equal to the
        ``lower`` bound has multiples of the interval width added to it,
        until the result lies in the desirable interval.

        Each ``v`` in ``values`` that starts greater than the ``upper``
        bound has multiples of the interval width subtracted from it,
        until the result lies in the desired interval.
        """
        translated_values = (
            numeric_values.values.copy()
            if isinstance(numeric_values, xr.DataArray)
            else numeric_values.copy()
        )

        interval_width = upper - lower
        if interval_width <= 0:
            raise ValueError(
                f"Upper bound ({upper}) must be strictly "
                f"greater than lower bound ({lower})"
            )

        while np.any(
            (translated_values <= lower) | (translated_values > upper)
        ):
            translated_values[translated_values <= lower] += interval_width
            translated_values[translated_values > upper] -= interval_width

        if isinstance(numeric_values, xr.DataArray):
            translated_values = numeric_values.copy(
                deep=True, data=translated_values
            )
        return translated_values

    @pytest.fixture
    def spinning_on_the_spot(self) -> xr.DataArray:
        """Simulate data for an individual's head spinning on the spot.

        The left / right keypoints move in a circular motion counter-clockwise
        around the unit circle centred on the origin, always opposite each
        other.
        The left keypoint starts on the negative x-axis, and the motion is
        split into 8 time points of uniform rotation angles.
        """
        data = np.zeros(shape=(8, 2, 2), dtype=float)
        data[:, :, 0] = np.array(
            [
                -self.x_axis,
                (-self.x_axis - self.y_axis) / self.sqrt_2,
                -self.y_axis,
                (self.x_axis - self.y_axis) / self.sqrt_2,
                self.x_axis,
                (self.x_axis + self.y_axis) / self.sqrt_2,
                self.y_axis,
                (-self.x_axis + self.y_axis) / self.sqrt_2,
            ]
        )
        data[:, :, 1] = -data[:, :, 0]
        return xr.DataArray(
            data=data,
            dims=["time", "space", "keypoints"],
            coords={"space": ["x", "y"], "keypoints": ["left", "right"]},
        )

    @pytest.mark.parametrize(
        ["swap_left_right", "swap_camera_angle", "swap_angle_rotation"],
        [
            pytest.param(True, True, True, id="(TTT) LR, Camera, Angle"),
            pytest.param(True, True, False, id="(TTF) LR, Camera"),
            pytest.param(True, False, True, id="(TFT) LR, Angle"),
            pytest.param(True, False, False, id="(TFF) LR"),
            pytest.param(False, True, True, id="(FTT) Camera, Angle"),
            pytest.param(False, True, False, id="(FTF) Camera"),
            pytest.param(False, False, True, id="(FFT) Angle"),
            pytest.param(False, False, False, id="(FFF)"),
        ],
    )
    def test_antisymmetry_properties(
        self,
        spinning_on_the_spot: xr.DataArray,
        swap_left_right: bool,
        swap_camera_angle: bool,
        swap_angle_rotation: bool,
    ) -> None:
        r"""Test antisymmetry arises where expected.

        Reversing the right and left keypoints, or the camera position, has the
        effect of mapping angles to the "opposite side" of the unit circle.
        Explicitly;
        - :math:`\theta <= 0` is mapped to :math:`\theta + 180`,
        - :math:`\theta > 0` is mapped to :math:`\theta - 180`.

        In theory, the antisymmetry of ``angle_rotates`` should be covered by
        the underlying tests for ``compute_signed_angle_2d``, however we
        include this case here for additional checks in conjunction with other
        behaviour.
        """
        reference_vector = self.x_axis
        left_keypoint = "left"
        right_keypoint = "right"

        args_to_function = {}
        if swap_left_right:
            args_to_function["left_keypoint"] = right_keypoint
            args_to_function["right_keypoint"] = left_keypoint
        else:
            args_to_function["left_keypoint"] = left_keypoint
            args_to_function["right_keypoint"] = right_keypoint
        if swap_camera_angle:
            args_to_function["camera_view"] = "bottom_up"
        if swap_angle_rotation:
            args_to_function["angle_rotates"] = "forward to ref"

        # mypy call here is angry, https://github.com/python/mypy/issues/1969
        with_orientations_swapped = kinematics.compute_forward_vector_angle(
            data=spinning_on_the_spot,
            reference_vector=reference_vector,
            **args_to_function,  # type: ignore[arg-type]
        )
        without_orientations_swapped = kinematics.compute_forward_vector_angle(
            data=spinning_on_the_spot,
            left_keypoint=left_keypoint,
            right_keypoint=right_keypoint,
            reference_vector=reference_vector,
        )

        expected_orientations = without_orientations_swapped.copy(deep=True)
        if swap_left_right:
            expected_orientations = self.push_into_range(
                expected_orientations + 180.0
            )
        if swap_camera_angle:
            expected_orientations = self.push_into_range(
                expected_orientations + 180.0
            )
        if swap_angle_rotation:
            expected_orientations *= -1.0
        expected_orientations = self.push_into_range(expected_orientations)

        xr.testing.assert_allclose(
            with_orientations_swapped, expected_orientations
        )
