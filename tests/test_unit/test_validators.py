from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from movement.io.validators import (
    ValidFile,
    ValidHDF5,
    ValidPosesCSV,
    ValidPoseTracks,
)


class TestValidators:
    """Test suite for the validators module."""

    pose_tracks = [
        {
            "names": None,
            "array_type": "multi_track_array",
            "individual_names_expected_exception": does_not_raise(
                ["individual_0", "individual_1"]
            ),
            "keypoint_names_expected_exception": does_not_raise(
                ["keypoint_0", "keypoint_1"]
            ),
        },  # valid input, will generate default names
        {
            "names": ["a", "b"],
            "array_type": "multi_track_array",
            "individual_names_expected_exception": does_not_raise(["a", "b"]),
            "keypoint_names_expected_exception": does_not_raise(["a", "b"]),
        },  # valid input
        {
            "names": ("a", "b"),
            "array_type": "multi_track_array",
            "individual_names_expected_exception": does_not_raise(["a", "b"]),
            "keypoint_names_expected_exception": does_not_raise(["a", "b"]),
        },  # valid input, will be converted to ["a", "b"]
        {
            "names": [1, 2],
            "array_type": "multi_track_array",
            "individual_names_expected_exception": does_not_raise(["1", "2"]),
            "keypoint_names_expected_exception": does_not_raise(["1", "2"]),
        },  # valid input, will be converted to ["1", "2"]
        {
            "names": "a",
            "array_type": "single_track_array",
            "individual_names_expected_exception": does_not_raise(["a"]),
            "keypoint_names_expected_exception": pytest.raises(ValueError),
        },  # single track array with multiple keypoints
        {
            "names": "a",
            "array_type": "single_keypoint_array",
            "individual_names_expected_exception": pytest.raises(ValueError),
            "keypoint_names_expected_exception": does_not_raise(["a"]),
        },  # single keypoint array with multiple tracks
        {
            "names": 5,
            "array_type": "multi_track_array",
            "individual_names_expected_exception": pytest.raises(ValueError),
            "keypoint_names_expected_exception": pytest.raises(ValueError),
        },  # invalid input
    ]

    @pytest.fixture(params=pose_tracks)
    def pose_tracks_params(self, request):
        """Return a dictionary containing parameters for testing
        pose track keypoint and individual names."""
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
    def test_ValidFile(self, invalid_input, expected_exception, request):
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
    def test_ValidHDF5(self, invalid_input, expected_exception, request):
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
            ("invalid_single_animal_csv_file", pytest.raises(ValueError)),
            ("invalid_multi_animal_csv_file", pytest.raises(ValueError)),
        ],
    )
    def test_ValidPosesCSV(self, invalid_input, expected_exception, request):
        """Test that invalid CSV files raise the appropriate errors."""
        file_path = request.getfixturevalue(invalid_input)
        with expected_exception:
            ValidPosesCSV(file_path)

    @pytest.mark.parametrize(
        "invalid_tracks_array",
        [
            None,  # invalid, argument is non-optional
            [1, 2, 3],  # not an ndarray
            np.zeros((10, 2, 3)),  # not 4d
            np.zeros((10, 2, 3, 4)),  # last dim not 2 or 3
        ],
    )
    def test_ValidPoseTrack_tracks(self, invalid_tracks_array):
        """Test that invalid tracks arrays raise the appropriate errors."""
        with pytest.raises(ValueError):
            ValidPoseTracks(tracks_array=invalid_tracks_array)

    @pytest.mark.parametrize(
        "scores_array, expected_exception",
        [
            (
                np.ones((10, 3, 2)),
                pytest.raises(ValueError),
            ),  # will not match tracks_array shape
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
    def test_ValidPoseTrack_scores(
        self,
        scores_array,
        expected_exception,
        valid_tracks_array,
    ):
        """Test that invalid scores arrays raise the appropriate errors."""
        with expected_exception:
            poses = ValidPoseTracks(
                tracks_array=valid_tracks_array("multi_track_array"),
                scores_array=scores_array,
            )
            if scores_array is None:
                assert np.all(np.isnan(poses.scores_array))

    def test_ValidPoseTrack_keypoint_names(
        self, pose_tracks_params, valid_tracks_array
    ):
        """Test that invalid keypoint names raise the appropriate errors."""
        with pose_tracks_params.get("keypoint_names_expected_exception") as e:
            poses = ValidPoseTracks(
                tracks_array=valid_tracks_array(
                    pose_tracks_params.get("array_type")
                ),
                keypoint_names=pose_tracks_params.get("names"),
            )
            assert poses.keypoint_names == e

    def test_ValidPoseTrack_individual_names(
        self, pose_tracks_params, valid_tracks_array
    ):
        """Test that invalid keypoint names raise the appropriate errors."""
        with pose_tracks_params.get(
            "individual_names_expected_exception"
        ) as e:
            poses = ValidPoseTracks(
                tracks_array=valid_tracks_array(
                    pose_tracks_params.get("array_type")
                ),
                individual_names=pose_tracks_params.get("names"),
            )
            assert poses.individual_names == e
