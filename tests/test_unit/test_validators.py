from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from movement.validators.datasets import ValidBboxesDataset, ValidPosesDataset
from movement.validators.files import ValidDeepLabCutCSV, ValidFile, ValidHDF5


class TestValidators:
    """Test suite for the validators module."""

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
    def position_array_params(self, request):
        """Return a dictionary containing parameters for testing
        position array keypoint and individual names.
        """
        return request.param

    @pytest.fixture
    def valid_bboxes_inputs(self):
        """Return a dictionary with valid inputs for a ValidBboxesDataset."""
        n_frames, n_individuals, n_space = (10, 2, 2)
        # valid array for position or shape
        valid_bbox_array = np.zeros((n_frames, n_individuals, n_space))

        return {
            "position_array": valid_bbox_array,
            "shape_array": valid_bbox_array,
            "individual_names": [
                "id_" + str(id) for id in range(valid_bbox_array.shape[1])
            ],
        }

    @pytest.mark.parametrize(
        "invalid_input, expected_exception",
        [
            ("unreadable_file", pytest.raises(PermissionError)),
            ("unwriteable_file", pytest.raises(PermissionError)),
            ("fake_h5_file", pytest.raises(FileExistsError)),
            ("wrong_ext_file", pytest.raises(ValueError)),
            ("nonexistent_file", pytest.raises(FileNotFoundError)),
            ("directory", pytest.raises(IsADirectoryError)),
        ],
    )
    def test_file_validator_with_invalid_input(
        self, invalid_input, expected_exception, request
    ):
        """Test that invalid files raise the appropriate errors."""
        invalid_dict = request.getfixturevalue(invalid_input)
        with expected_exception:
            ValidFile(
                invalid_dict.get("file_path"),
                expected_permission=invalid_dict.get("expected_permission"),
                expected_suffix=invalid_dict.get("expected_suffix", []),
            )

    @pytest.mark.parametrize(
        "invalid_input, expected_exception",
        [
            ("h5_file_no_dataframe", pytest.raises(ValueError)),
            ("fake_h5_file", pytest.raises(ValueError)),
        ],
    )
    def test_hdf5_validator_with_invalid_input(
        self, invalid_input, expected_exception, request
    ):
        """Test that invalid HDF5 files raise the appropriate errors."""
        invalid_dict = request.getfixturevalue(invalid_input)
        with expected_exception:
            ValidHDF5(
                invalid_dict.get("file_path"),
                expected_datasets=invalid_dict.get("expected_datasets"),
            )

    @pytest.mark.parametrize(
        "invalid_input, expected_exception",
        [
            ("invalid_single_individual_csv_file", pytest.raises(ValueError)),
            ("invalid_multi_individual_csv_file", pytest.raises(ValueError)),
        ],
    )
    def test_poses_csv_validator_with_invalid_input(
        self, invalid_input, expected_exception, request
    ):
        """Test that invalid CSV files raise the appropriate errors."""
        file_path = request.getfixturevalue(invalid_input)
        with expected_exception:
            ValidDeepLabCutCSV(file_path)

    @pytest.mark.parametrize(
        "invalid_position_array",
        [
            None,  # invalid, argument is non-optional
            [1, 2, 3],  # not an ndarray
            np.zeros((10, 2, 3)),  # not 4d
            np.zeros((10, 2, 3, 4)),  # last dim not 2 or 3
        ],
    )
    def test_poses_dataset_validator_with_invalid_position_array(
        self, invalid_position_array
    ):
        """Test that invalid position arrays raise the appropriate errors."""
        with pytest.raises(ValueError):
            ValidPosesDataset(position_array=invalid_position_array)

    @pytest.mark.parametrize(
        "confidence_array, expected_exception",
        [
            (
                np.ones((10, 3, 2)),
                pytest.raises(ValueError),
            ),  # will not match position_array shape
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
        self,
        confidence_array,
        expected_exception,
        valid_position_array,
    ):
        """Test that invalid confidence arrays raise the appropriate errors."""
        with expected_exception:
            poses = ValidPosesDataset(
                position_array=valid_position_array("multi_individual_array"),
                confidence_array=confidence_array,
            )
            if confidence_array is None:
                assert np.all(np.isnan(poses.confidence_array))

    def test_poses_dataset_validator_keypoint_names(
        self, position_array_params, valid_position_array
    ):
        """Test that invalid keypoint names raise the appropriate errors."""
        with position_array_params.get(
            "keypoint_names_expected_exception"
        ) as e:
            poses = ValidPosesDataset(
                position_array=valid_position_array(
                    position_array_params.get("array_type")
                ),
                keypoint_names=position_array_params.get("names"),
            )
            assert poses.keypoint_names == e

    def test_poses_dataset_validator_individual_names(
        self, position_array_params, valid_position_array
    ):
        """Test that invalid keypoint names raise the appropriate errors."""
        with position_array_params.get(
            "individual_names_expected_exception"
        ) as e:
            poses = ValidPosesDataset(
                position_array=valid_position_array(
                    position_array_params.get("array_type")
                ),
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
        self, valid_position_array, source_software, expected_exception
    ):
        """Test that the source_software attribute is validated properly.
        LightnigPose is incompatible with multi-individual arrays.
        """
        with expected_exception:
            ds = ValidPosesDataset(
                position_array=valid_position_array("multi_individual_array"),
                source_software=source_software,
            )

            if source_software is not None:
                assert ds.source_software == source_software
            else:
                assert ds.source_software is None

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
                "Expected 'position_array' to have 2 spatial "
                "coordinates, but got 3.",
            ),  # last dim not 2
        ],
    )
    def test_bboxes_dataset_validator_with_invalid_position_array(
        self, invalid_position_array, log_message, request
    ):
        """Test that invalid centroid position arrays raise an error."""
        with pytest.raises(ValueError) as excinfo:
            ValidBboxesDataset(
                position_array=invalid_position_array,
                shape_array=request.getfixturevalue("valid_bboxes_inputs")[
                    "shape_array"
                ],
                individual_names=request.getfixturevalue(
                    "valid_bboxes_inputs"
                )["individual_names"],
            )
        assert str(excinfo.value) == log_message

    @pytest.mark.parametrize(
        "invalid_shape_array, log_message",
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
                "Expected 'shape_array' to have 2 spatial "
                "coordinates, but got 3.",
            ),  # last dim (spatial) not 2
        ],
    )
    def test_bboxes_dataset_validator_with_invalid_shape_array(
        self, invalid_shape_array, log_message, request
    ):
        """Test that invalid shape arrays raise an error."""
        with pytest.raises(ValueError) as excinfo:
            ValidBboxesDataset(
                position_array=request.getfixturevalue("valid_bboxes_inputs")[
                    "position_array"
                ],
                shape_array=invalid_shape_array,
                individual_names=request.getfixturevalue(
                    "valid_bboxes_inputs"
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
                "Expected 'individual_names' to have length 2, but got 3.",
            ),  # length doesn't match position_array.shape[1]
            (
                ["id_1", "id_1"],
                pytest.raises(ValueError),
                "individual_names passed to the dataset are not unique. "
                "There are 2 elements in the list, but "
                "only 1 are unique.",
            ),  # some IDs are not unique
        ],
    )
    def test_bboxes_dataset_validator_individual_names(
        self, list_individual_names, expected_exception, log_message, request
    ):
        """Test individual_names inputs."""
        with expected_exception as excinfo:
            ds = ValidBboxesDataset(
                position_array=request.getfixturevalue("valid_bboxes_inputs")[
                    "position_array"
                ],
                shape_array=request.getfixturevalue("valid_bboxes_inputs")[
                    "shape_array"
                ],
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
                "Expected 'confidence_array' to have shape (10, 2), "
                "but got (10, 3, 2).",
            ),  # will not match position_array shape
            (
                [1, 2, 3],
                pytest.raises(ValueError),
                f"Expected a numpy array, but got {type(list())}.",
            ),  # not an ndarray, should raise ValueError
            (
                None,
                does_not_raise(),
                "",
            ),  # valid, should default to array of NaNs
        ],
    )
    def test_bboxes_dataset_validator_confidence_array(
        self, confidence_array, expected_exception, log_message, request
    ):
        """Test that invalid confidence arrays raise the appropriate errors."""
        with expected_exception as excinfo:
            poses = ValidBboxesDataset(
                position_array=request.getfixturevalue("valid_bboxes_inputs")[
                    "position_array"
                ],
                shape_array=request.getfixturevalue("valid_bboxes_inputs")[
                    "shape_array"
                ],
                individual_names=request.getfixturevalue(
                    "valid_bboxes_inputs"
                )["individual_names"],
                confidence_array=confidence_array,
            )
        if confidence_array is None:
            assert np.all(
                np.isnan(poses.confidence_array)
            )  # assert it is a NaN array
            assert (
                poses.confidence_array.shape == poses.position_array.shape[:-1]
            )  # assert shape matches position array
        else:
            assert str(excinfo.value) == log_message
