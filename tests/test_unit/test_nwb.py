import datetime

import ndx_pose
import numpy as np
import pytest
from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

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
        ds.sel(individuals="individual1"),
        subject="individual1",
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
    assert len(pose_estimation[0].pose_estimation_series) == 12

    # Assert the name of the first PoseEstimationSeries
    assert "snout" in pose_estimation[0].pose_estimation_series

    # Assert the name of the Skeleton
    assert "individual1_skeleton" in skeletons.skeletons


def create_test_pose_estimation_series(
    n_time=100, n_dims=2, keypoint="front_left_paw"
):
    data = np.random.rand(
        n_time, n_dims
    )  # num_frames x (x, y) but can be (x, y, z)
    timestamps = np.linspace(0, 10, num=n_time)  # a timestamp for every frame
    confidence = np.ones((n_time,))  # a confidence value for every frame
    reference_frame = "(0,0,0) corresponds to ..."
    confidence_definition = "Softmax output of the deep neural network."

    return PoseEstimationSeries(
        name=keypoint,
        description="Marker placed around fingers of front left paw.",
        data=data,
        unit="pixels",
        reference_frame=reference_frame,
        timestamps=timestamps,
        confidence=confidence,
        confidence_definition=confidence_definition,
    )


@pytest.mark.parametrize(
    "n_time, n_dims",
    [
        (100, 2),  # 2D data
        (50, 3),  # 3D data
    ],
)
def test_convert_pose_estimation_series(n_time, n_dims):
    # Create a sample PoseEstimationSeries object
    pose_estimation_series = create_test_pose_estimation_series(
        n_time=n_time, n_dims=n_dims, keypoint="leftear"
    )

    # Call the function
    movement_dataset = _convert_pose_estimation_series(
        pose_estimation_series,
        keypoint="leftear",
        subject_name="individual1",
        source_software="software1",
        source_file="file1",
    )

    # Assert the dimensions of the movement dataset
    assert movement_dataset.sizes == {
        "time": n_time,
        "individuals": 1,
        "keypoints": 1,
        "space": n_dims,
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
    ds = sample_data.fetch_dataset("DLC_two-mice.predictions.csv")
    session_start_time = datetime.datetime.now(datetime.timezone.utc)
    nwbfile_individual1 = NWBFile(
        session_description="session_description",
        identifier="individual1",
        session_start_time=session_start_time,
    )
    add_movement_dataset_to_nwb(
        nwbfile_individual1, ds.sel(individuals=["individual1"])
    )
    assert (
        "PoseEstimation"
        in nwbfile_individual1.processing["behavior"].data_interfaces
    )
    assert (
        "Skeletons"
        in nwbfile_individual1.processing["behavior"].data_interfaces
    )


def test_add_movement_dataset_to_nwb_multiple_files():
    ds = sample_data.fetch_dataset("DLC_two-mice.predictions.csv")
    session_start_time = datetime.datetime.now(datetime.timezone.utc)
    nwbfile_individual1 = NWBFile(
        session_description="session_description",
        identifier="individual1",
        session_start_time=session_start_time,
    )
    nwbfile_individual2 = NWBFile(
        session_description="session_description",
        identifier="individual2",
        session_start_time=session_start_time,
    )

    nwbfiles = [nwbfile_individual1, nwbfile_individual2]
    add_movement_dataset_to_nwb(nwbfiles, ds)


def create_test_pose_nwb(identifier="subject1", write_to_disk=False):
    # initialize an NWBFile object
    nwbfile = NWBFile(
        session_description="session_description",
        identifier=identifier,
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )

    # add a subject to the NWB file
    subject = Subject(subject_id=identifier, species="Mus musculus")
    nwbfile.subject = subject

    skeleton = Skeleton(
        name="subject1_skeleton",
        nodes=["front_left_paw", "body", "front_right_paw"],
        edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
        subject=subject,
    )

    skeletons = Skeletons(skeletons=[skeleton])

    # create a device for the camera
    camera1 = nwbfile.create_device(
        name="camera1",
        description="camera for recording behavior",
        manufacturer="my manufacturer",
    )

    n_time = 100
    n_dims = 2  # 2D data
    front_left_paw = create_test_pose_estimation_series(
        n_time=n_time, n_dims=n_dims, keypoint="front_left_paw"
    )

    body = create_test_pose_estimation_series(
        n_time=n_time, n_dims=n_dims, keypoint="body"
    )
    front_right_paw = create_test_pose_estimation_series(
        n_time=n_time, n_dims=n_dims, keypoint="front_right_paw"
    )

    # store all PoseEstimationSeries in a list
    pose_estimation_series = [front_left_paw, body, front_right_paw]

    pose_estimation = PoseEstimation(
        name="PoseEstimation",
        pose_estimation_series=pose_estimation_series,
        description=(
            "Estimated positions of front paws of subject1 using DeepLabCut."
        ),
        original_videos=["path/to/camera1.mp4"],
        labeled_videos=["path/to/camera1_labeled.mp4"],
        dimensions=np.array(
            [[640, 480]], dtype="uint16"
        ),  # pixel dimensions of the video
        devices=[camera1],
        scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
        source_software="DeepLabCut",
        source_software_version="2.3.8",
        skeleton=skeleton,  # link to the skeleton object
    )

    behavior_pm = nwbfile.create_processing_module(
        name="behavior",
        description="processed behavioral data",
    )
    behavior_pm.add(skeletons)
    behavior_pm.add(pose_estimation)

    # write the NWBFile to disk
    if write_to_disk:
        path = "test_pose.nwb"
        with NWBHDF5IO(path, mode="w") as io:
            io.write(nwbfile)
    else:
        return nwbfile


def test_convert_nwb_to_movement():
    create_test_pose_nwb(write_to_disk=True)
    nwb_filepaths = ["test_pose.nwb"]
    movement_dataset = convert_nwb_to_movement(nwb_filepaths)

    assert movement_dataset.sizes == {
        "time": 100,
        "individuals": 1,
        "keypoints": 3,
        "space": 2,
    }
