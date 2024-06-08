import ndx_pose
import numpy as np
import pynwb
import pytest
import xarray as xr

from movement import sample_data
from movement.io.nwb import (
    _convert_pose_estimation_series,
    _create_pose_and_skeleton_objects,
    add_movement_dataset_to_nwb,
    convert_nwb_to_movement,
)


def test_create_pose_and_skeleton_objects():
    # Create a sample dataset
    ds = sample_data.fetch_dataset("DLC_two-mice.predictions.csv")

    # Call the function
    pose_estimation, skeletons = _create_pose_and_skeleton_objects(
        ds,
        subject="subject1",
        pose_estimation_series_kwargs=None,
        pose_estimation_kwargs=None,
        skeleton_kwargs=None,
    )

    # Assert the output types
    assert isinstance(pose_estimation, list)
    assert isinstance(skeletons, ndx_pose.Skeletons)

    # Assert the length of pose_estimation list
    assert len(pose_estimation) == 1

    # Assert the length of pose_estimation_series list
    assert len(pose_estimation[0].pose_estimation_series) == 2

    # Assert the name of the first PoseEstimationSeries
    assert pose_estimation[0].pose_estimation_series[0].name == "keypoint1"

    # Assert the name of the second PoseEstimationSeries
    assert pose_estimation[0].pose_estimation_series[1].name == "keypoint2"

    # Assert the name of the Skeleton
    assert skeletons.skeletons[0].name == "subject1_skeleton"


def test__convert_pose_estimation_series():
    # Create a sample PoseEstimationSeries object
    pose_estimation_series = ndx_pose.PoseEstimationSeries(
        name="keypoint1",
        data=np.random.rand(10, 3),
        confidence=np.random.rand(10),
        unit="pixels",
        timestamps=np.arange(10),
    )

    # Call the function
    movement_dataset = _convert_pose_estimation_series(
        pose_estimation_series,
        keypoint="keypoint1",
        subject_name="subject1",
        source_software="software1",
        source_file="file1",
    )

    # Assert the dimensions of the movement dataset
    assert movement_dataset.dims == {
        "time": 10,
        "individuals": 1,
        "keypoints": 1,
        "space": 3,
    }

    # Assert the values of the position variable
    np.testing.assert_array_equal(
        movement_dataset["position"].values,
        pose_estimation_series.data[:, np.newaxis, np.newaxis, :],
    )

    # Assert the values of the confidence variable
    np.testing.assert_array_equal(
        movement_dataset["confidence"].values,
        pose_estimation_series.confidence[:, np.newaxis, np.newaxis],
    )

    # Assert the attributes of the movement dataset
    assert movement_dataset.attrs == {
        "fps": np.nanmedian(1 / np.diff(pose_estimation_series.timestamps)),
        "time_units": pose_estimation_series.timestamps_unit,
        "source_software": "software1",
        "source_file": "file1",
    }


def test_add_movement_dataset_to_nwb_single_file():
    # Create a sample NWBFile
    nwbfile = pynwb.NWBFile(
        "session_description", "identifier", "session_start_time"
    )
    # Create a sample movement dataset
    movement_dataset = xr.Dataset(
        {
            "keypoints": (["keypoints"], ["keypoint1", "keypoint2"]),
            "position": (["time", "keypoints"], [[1, 2], [3, 4]]),
            "confidence": (["time", "keypoints"], [[0.9, 0.8], [0.7, 0.6]]),
            "time": [0, 1],
            "individuals": ["subject1"],
        }
    )
    # Call the function
    add_movement_dataset_to_nwb(nwbfile, movement_dataset)
    # Assert the presence of PoseEstimation and Skeletons in the NWBFile
    assert "PoseEstimation" in nwbfile.processing["behavior"]
    assert "Skeletons" in nwbfile.processing["behavior"]


def test_add_movement_dataset_to_nwb_multiple_files():
    # Create sample NWBFiles
    nwbfiles = [
        pynwb.NWBFile(
            "session_description1", "identifier1", "session_start_time1"
        ),
        pynwb.NWBFile(
            "session_description2", "identifier2", "session_start_time2"
        ),
    ]
    # Create a sample movement dataset
    movement_dataset = xr.Dataset(
        {
            "keypoints": (["keypoints"], ["keypoint1", "keypoint2"]),
            "position": (["time", "keypoints"], [[1, 2], [3, 4]]),
            "confidence": (["time", "keypoints"], [[0.9, 0.8], [0.7, 0.6]]),
            "time": [0, 1],
            "individuals": ["subject1", "subject2"],
        }
    )
    # Call the function
    add_movement_dataset_to_nwb(nwbfiles, movement_dataset)
    # Assert the presence of PoseEstimation and Skeletons in each NWBFile
    for nwbfile in nwbfiles:
        assert "PoseEstimation" in nwbfile.processing["behavior"]
        assert "Skeletons" in nwbfile.processing["behavior"]


def test_convert_nwb_to_movement():
    # Create sample NWB files
    nwb_filepaths = [
        "/path/to/file1.nwb",
        "/path/to/file2.nwb",
        "/path/to/file3.nwb",
    ]
    pose_estimation_series = {
        "keypoint1": ndx_pose.PoseEstimationSeries(
            name="keypoint1",
            data=np.random.rand(10, 3),
            confidence=np.random.rand(10),
            unit="pixels",
            timestamps=np.arange(10),
        ),
        "keypoint2": ndx_pose.PoseEstimationSeries(
            name="keypoint2",
            data=np.random.rand(10, 3),
            confidence=np.random.rand(10),
            unit="pixels",
            timestamps=np.arange(10),
        ),
    }

    # Mock the NWBHDF5IO read method
    def mock_read(filepath):
        nwbfile = pynwb.NWBFile(
            "session_description", "identifier", "session_start_time"
        )

        pose_estimation = ndx_pose.PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=pose_estimation_series,
            description="Pose estimation data",
            source_software="software1",
            skeleton=ndx_pose.Skeleton(
                name="skeleton1", nodes=["node1", "node2"]
            ),
        )
        behavior_pm = pynwb.ProcessingModule(
            name="behavior", description="Behavior data"
        )
        behavior_pm.add(pose_estimation)
        nwbfile.add_processing_module(behavior_pm)
        return nwbfile

    # Patch the NWBHDF5IO read method with the mock
    with pytest.patch("pynwb.NWBHDF5IO.read", side_effect=mock_read):
        # Call the function
        movement_dataset = convert_nwb_to_movement(nwb_filepaths)

    # Assert the dimensions of the movement dataset
    assert movement_dataset.dims == {
        "time": 10,
        "individuals": 3,
        "keypoints": 2,
        "space": 3,
    }

    # Assert the values of the position variable
    np.testing.assert_array_equal(
        movement_dataset["position"].values,
        np.concatenate(
            [
                pose_estimation_series["keypoint1"].data[
                    :, np.newaxis, np.newaxis, :
                ],
                pose_estimation_series["keypoint2"].data[
                    :, np.newaxis, np.newaxis, :
                ],
            ],
            axis=1,
        ),
    )

    # Assert the values of the confidence variable
    np.testing.assert_array_equal(
        movement_dataset["confidence"].values,
        np.concatenate(
            [
                pose_estimation_series["keypoint1"].confidence[
                    :, np.newaxis, np.newaxis
                ],
                pose_estimation_series["keypoint2"].confidence[
                    :, np.newaxis, np.newaxis
                ],
            ],
            axis=1,
        ),
    )

    # Assert the attributes of the movement dataset
    assert movement_dataset.attrs == {
        "fps": np.nanmedian(
            1 / np.diff(pose_estimation_series["keypoint1"].timestamps)
        ),
        "time_units": pose_estimation_series["keypoint1"].timestamps_unit,
        "source_software": "software1",
        "source_file": None,
    }
