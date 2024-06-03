from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from movement.io.validators import (
    ValidBboxesDataset,
    ValidDeepLabCutCSV,
    ValidFile,
    ValidHDF5,
    ValidPosesDataset,
)


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
        "invalid_centroid_position_array",
        [
            None,  # invalid, argument is non-optional
            [1, 2, 3],  # not an ndarray
            np.zeros((10, 2)),  # not 3d
            np.zeros((10, 2, 3)),  # last dim not 2
        ],
    )
    def test_bboxes_dataset_validator_with_invalid_centroid_position_array(
        self, invalid_centroid_position_array
    ):
        """Test that invalid centroid position arrays raise an error."""
        with pytest.raises(ValueError):
            ValidBboxesDataset(
                centroid_position_array=invalid_centroid_position_array,
                shape_array=np.zeros((10, 2, 2)),
                IDs=["id_" + str(id) for id in [1, 2, 3, 4]],
            )

    @pytest.mark.parametrize(
        "invalid_shape_array",
        [
            None,  # invalid, argument is non-optional
            [1, 2, 3],  # not an ndarray
            np.zeros((10, 2)),  # not 3d
            np.zeros((10, 2, 3)),  # last dim not 2
        ],
    )
    def test_bboxes_dataset_validator_with_invalid_shape_array(
        self, invalid_shape_array
    ):
        """Test that invalid shape arrays raise an error."""
        with pytest.raises(ValueError):
            ValidBboxesDataset(
                centroid_position_array=np.zeros((10, 2, 2)),
                shape_array=invalid_shape_array,
                IDs=["id_" + str(id) for id in [1, 2, 3, 4]],
            )

    @pytest.mark.parametrize(
        "invalid_ID_list",
        [
            None,  # invalid, argument is non-optional
            [1, 2, 3],  # length doesn't match centroid_position_array.shape[1]
            [1, 2],  # IDs not in the expected format
            ["id_0", "id_2"],  # some IDs are not 1-based
            [1, 1, 2, 3],  # some IDs are not unique
        ],
    )
    def test_bboxes_dataset_validator_with_invalid_ID_array(
        self, invalid_ID_list
    ):
        """Test that invalid ID arrays raise an error."""
        # TODO: can I check the error is raised where I expect it?
        with pytest.raises(ValueError):
            ValidBboxesDataset(
                centroid_position_array=np.zeros(
                    (10, 2, 2)
                ),  #  (n_frames, n_unique_IDs, n_space)
                shape_array=np.zeros((10, 2, 2)),
                IDs=invalid_ID_list,
            )

    # def test_bboxes_dataset_validator_confidence_array():
    #     pass
