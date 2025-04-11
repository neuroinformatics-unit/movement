from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from movement.validators.datasets import ValidBboxesDataset, ValidPosesDataset

position_arrays = [
    {
        "names": None,
        "array_type": "multi_individual_array",
        "individual_names_expected_exception": does_not_raise(
            ["individual_0", "individual_1"]
        ),
        "keypoint_names_expected_exception": does_not_raise(
            ["keypoint_0", "keypoint_1"]
        ),
    },  # valid input, will generate default names
    {
        "names": ["a", "b"],
        "array_type": "multi_individual_array",
        "individual_names_expected_exception": does_not_raise(["a", "b"]),
        "keypoint_names_expected_exception": does_not_raise(["a", "b"]),
    },  # valid input
    {
        "names": ("a", "b"),
        "array_type": "multi_individual_array",
        "individual_names_expected_exception": does_not_raise(["a", "b"]),
        "keypoint_names_expected_exception": does_not_raise(["a", "b"]),
    },  # valid input, will be converted to ["a", "b"]
    {
        "names": [1, 2],
        "array_type": "multi_individual_array",
        "individual_names_expected_exception": does_not_raise(["1", "2"]),
        "keypoint_names_expected_exception": does_not_raise(["1", "2"]),
    },  # valid input, will be converted to ["1", "2"]
    {
        "names": "a",
        "array_type": "single_individual_array",
        "individual_names_expected_exception": does_not_raise(["a"]),
        "keypoint_names_expected_exception": pytest.raises(ValueError),
    },  # single individual array with multiple keypoints
    {
        "names": "a",
        "array_type": "single_keypoint_array",
        "individual_names_expected_exception": pytest.raises(ValueError),
        "keypoint_names_expected_exception": does_not_raise(["a"]),
    },  # single keypoint array with multiple individuals
    {
        "names": 5,
        "array_type": "multi_individual_array",
        "individual_names_expected_exception": pytest.raises(ValueError),
        "keypoint_names_expected_exception": pytest.raises(ValueError),
    },  # invalid input
]


@pytest.fixture(params=position_arrays)
def position_array_params(request):
    """Return a dictionary containing parameters for testing
    position array keypoint and individual names.
    """
    return request.param


# Fixtures bbox dataset
invalid_bboxes_arrays_and_expected_log = {
    key: [
        (
            None,
            f"Expected a numpy array, but got {type(None)}.",
        ),  # invalid, argument is non-optional
        (
            [1, 2, 3],
            f"Expected a numpy array, but got {type(list())}.",
        ),  # not an ndarray
        (
            np.zeros((10, 3, 2)),
            f"Expected '{key}_array' to have 2 spatial "
            "coordinates, but got 3.",
        ),  # `space` dim (at idx 1) not 2
    ]
    for key in ["position", "shape"]
}


# Tests pose dataset
@pytest.mark.parametrize(
    "invalid_position_array, log_message",
    [
        (
            None,
            f"Expected a numpy array, but got {type(None)}.",
        ),  # invalid, argument is non-optional
        (
            [1, 2, 3],
            f"Expected a numpy array, but got {type(list())}.",
        ),  # not an ndarray
        (
            np.zeros((10, 2, 3)),
            "Expected 'position_array' to have 4 dimensions, but got 3.",
        ),  # not 4d
        (
            np.zeros((10, 4, 3, 2)),
            "Expected 'position_array' to have 2 or 3 "
            "spatial dimensions, but got 4.",
        ),  # `space` dim (at idx 1) not 2 or 3
    ],
)
def test_poses_dataset_validator_with_invalid_position_array(
    invalid_position_array, log_message
):
    """Test that invalid position arrays raise the appropriate errors."""
    with pytest.raises(ValueError) as excinfo:
        ValidPosesDataset(position_array=invalid_position_array)
    assert str(excinfo.value) == log_message


@pytest.mark.parametrize(
    "confidence_array, expected_exception",
    [
        (
            np.ones((10, 2, 2)),
            pytest.raises(ValueError),
        ),  # will not match position_array shape (10, 2, 3, 2)
        (
            [1, 2, 3],
            pytest.raises(ValueError),
        ),  # not an ndarray, should raise ValueError
        (
            None,
            does_not_raise(),
        ),  # valid, should default to array of NaNs
    ],
)
def test_poses_dataset_validator_confidence_array(
    confidence_array, expected_exception, valid_poses_arrays
):
    """Test that invalid confidence arrays raise the appropriate errors."""
    with expected_exception:
        poses = ValidPosesDataset(
            position_array=valid_poses_arrays("multi_individual_array")[
                "position"
            ],
            confidence_array=confidence_array,
        )
        if confidence_array is None:
            assert np.all(np.isnan(poses.confidence_array))


@pytest.mark.filterwarnings(
    "ignore:.*Converting to a list of length 1.:UserWarning"
)
def test_poses_dataset_validator_keypoint_names(
    position_array_params, valid_poses_arrays
):
    """Test that invalid keypoint names raise the appropriate errors."""
    with position_array_params.get("keypoint_names_expected_exception") as e:
        poses = ValidPosesDataset(
            position_array=valid_poses_arrays(
                position_array_params.get("array_type")
            )["position"][:, :, :2, :],  # select up to the first 2 keypoints
            keypoint_names=position_array_params.get("names"),
        )
        assert poses.keypoint_names == e


@pytest.mark.filterwarnings(
    "ignore:.*Converting to a list of length 1.:UserWarning"
)
def test_poses_dataset_validator_individual_names(
    position_array_params, valid_poses_arrays
):
    """Test that invalid keypoint names raise the appropriate errors."""
    with position_array_params.get("individual_names_expected_exception") as e:
        poses = ValidPosesDataset(
            position_array=valid_poses_arrays(
                position_array_params.get("array_type")
            )["position"],
            individual_names=position_array_params.get("names"),
        )
        assert poses.individual_names == e


@pytest.mark.parametrize(
    "source_software, expected_exception",
    [
        (None, does_not_raise()),
        ("SLEAP", does_not_raise()),
        ("DeepLabCut", does_not_raise()),
        ("LightningPose", pytest.raises(ValueError)),
        ("fake_software", does_not_raise()),
        (5, pytest.raises(TypeError)),  # not a string
    ],
)
def test_poses_dataset_validator_source_software(
    valid_poses_arrays, source_software, expected_exception
):
    """Test that the source_software attribute is validated properly.
    LightnigPose is incompatible with multi-individual arrays.
    """
    with expected_exception:
        ds = ValidPosesDataset(
            position_array=valid_poses_arrays("multi_individual_array")[
                "position"
            ],
            source_software=source_software,
        )

        if source_software is not None:
            assert ds.source_software == source_software
        else:
            assert ds.source_software is None


# Tests bboxes dataset
@pytest.mark.parametrize(
    "invalid_position_array, log_message",
    invalid_bboxes_arrays_and_expected_log["position"],
)
def test_bboxes_dataset_validator_with_invalid_position_array(
    invalid_position_array, log_message, request
):
    """Test that invalid centroid position arrays raise an error."""
    with pytest.raises(ValueError) as excinfo:
        ValidBboxesDataset(
            position_array=invalid_position_array,
            shape_array=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["shape"],
            individual_names=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["individual_names"],
        )
    assert str(excinfo.value) == log_message


@pytest.mark.parametrize(
    "invalid_shape_array, log_message",
    invalid_bboxes_arrays_and_expected_log["shape"],
)
def test_bboxes_dataset_validator_with_invalid_shape_array(
    invalid_shape_array, log_message, request
):
    """Test that invalid shape arrays raise an error."""
    with pytest.raises(ValueError) as excinfo:
        ValidBboxesDataset(
            position_array=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["position"],
            shape_array=invalid_shape_array,
            individual_names=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["individual_names"],
        )
    assert str(excinfo.value) == log_message


@pytest.mark.parametrize(
    "list_individual_names, expected_exception, log_message",
    [
        (
            None,
            does_not_raise(),
            "",
        ),  # valid, should default to unique IDs per frame
        (
            [1, 2, 3],
            pytest.raises(ValueError),
            "Expected 'individual_names' to have length 2, "
            f"but got {len([1, 2, 3])}.",
        ),  # length doesn't match position_array.shape[1]
        # from valid_bboxes_arrays_all_zeros fixture
        (
            ["id_1", "id_1"],
            pytest.raises(ValueError),
            "individual_names are not unique. "
            "There are 2 elements in the list, but "
            "only 1 are unique.",
        ),  # some IDs are not unique.
        # Note: length of individual_names list should match
        # n_individuals in valid_bboxes_arrays_all_zeros fixture
    ],
)
def test_bboxes_dataset_validator_individual_names(
    list_individual_names, expected_exception, log_message, request
):
    """Test individual_names inputs."""
    with expected_exception as excinfo:
        ds = ValidBboxesDataset(
            position_array=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["position"],
            shape_array=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["shape"],
            individual_names=list_individual_names,
        )
    if list_individual_names is None:
        # check IDs are unique per frame
        assert len(ds.individual_names) == len(set(ds.individual_names))
        assert ds.position_array.shape[1] == len(ds.individual_names)
    else:
        assert str(excinfo.value) == log_message


@pytest.mark.parametrize(
    "confidence_array, expected_exception, log_message",
    [
        (
            np.ones((10, 3, 2)),
            pytest.raises(ValueError),
            f"Expected 'confidence_array' to have shape (10, 2), "
            f"but got {np.ones((10, 3, 2)).shape}.",
        ),  # will not match shape of position_array in
        # valid_bboxes_arrays_all_zeros fixture
        (
            [1, 2, 3],
            pytest.raises(ValueError),
            f"Expected a numpy array, but got {type([1, 2, 3])}.",
        ),  # not an ndarray, should raise ValueError
        (
            None,
            does_not_raise(),
            "",
        ),  # valid, should default to array of NaNs
    ],
)
def test_bboxes_dataset_validator_confidence_array(
    confidence_array, expected_exception, log_message, request
):
    """Test that invalid confidence arrays raise the appropriate errors."""
    with expected_exception as excinfo:
        ds = ValidBboxesDataset(
            position_array=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["position"],
            shape_array=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["shape"],
            individual_names=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["individual_names"],
            confidence_array=confidence_array,
        )
    if confidence_array is None:
        assert np.all(
            np.isnan(ds.confidence_array)
        )  # assert it is a NaN array
        assert (
            ds.confidence_array.shape == ds.position_array.shape[:-1]
        )  # assert shape matches position array
    else:
        assert str(excinfo.value) == log_message


@pytest.mark.parametrize(
    "frame_array, expected_exception, log_message",
    [
        (
            np.arange(10).reshape(-1, 2),
            pytest.raises(ValueError),
            "Expected 'frame_array' to have shape (10, 1), but got (5, 2).",
        ),  # frame_array should be a column vector
        (
            [1, 2, 3],
            pytest.raises(ValueError),
            f"Expected a numpy array, but got {type(list())}.",
        ),  # not an ndarray, should raise ValueError
        (
            np.array([1, 10, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1),
            pytest.raises(ValueError),
            "Frame numbers in frame_array are not monotonically increasing.",
        ),
        (
            np.array([1, 2, 22, 23, 24, 25, 100, 101, 102, 103]).reshape(
                -1, 1
            ),
            does_not_raise(),
            "",
        ),  # valid, frame numbers are not continuous but are monotonically
        # increasing
        (
            None,
            does_not_raise(),
            "",
        ),  # valid, should return an array of frame numbers starting from 0
    ],
)
def test_bboxes_dataset_validator_frame_array(
    frame_array, expected_exception, log_message, request
):
    """Test that invalid frame arrays raise the appropriate errors."""
    with expected_exception as excinfo:
        ds = ValidBboxesDataset(
            position_array=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["position"],
            shape_array=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["shape"],
            individual_names=request.getfixturevalue(
                "valid_bboxes_arrays_all_zeros"
            )["individual_names"],
            frame_array=frame_array,
        )

    if frame_array is not None:
        assert str(getattr(excinfo, "value", "")) == log_message
    else:
        n_frames = ds.position_array.shape[0]
        default_frame_array = np.arange(n_frames).reshape(-1, 1)
        assert np.array_equal(ds.frame_array, default_frame_array)
        assert ds.frame_array.shape == (ds.position_array.shape[0], 1)
