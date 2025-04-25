import datetime

import ndx_pose
from pynwb import NWBFile

from movement import sample_data
from movement.io.nwb import _ds_to_pose_and_skeleton_objects
from movement.io.save_poses import to_nwb_file


def test_ds_to_pose_and_skeleton_objects():
    ds = sample_data.fetch_dataset("DLC_two-mice.predictions.csv")
    pose_estimation, skeletons = _ds_to_pose_and_skeleton_objects(
        ds.sel(individuals="individual1"),
        pose_estimation_series_kwargs=None,
        pose_estimation_kwargs=None,
        skeleton_kwargs=None,
    )
    # Assert the output types
    assert isinstance(pose_estimation, list)
    assert isinstance(skeletons, ndx_pose.Skeletons)
    # Assert the length of pose_estimation list (n_individuals)
    assert len(pose_estimation) == 1
    # Assert the length of pose_estimation_series list (n_keypoints)
    assert len(pose_estimation[0].pose_estimation_series) == 12
    # Assert the name of the first PoseEstimationSeries (first keypoint)
    assert "snout" in pose_estimation[0].pose_estimation_series
    # Assert the name of the Skeleton (individual1_skeleton)
    assert "individual1_skeleton" in skeletons.skeletons


def test_save_poses_to_single_nwb_file():
    ds = sample_data.fetch_dataset("DLC_two-mice.predictions.csv")
    session_start_time = datetime.datetime.now(datetime.UTC)
    nwbfile_individual1 = NWBFile(
        session_description="session_description",
        identifier="individual1",
        session_start_time=session_start_time,
    )
    to_nwb_file(ds.sel(individuals=["individual1"]), nwbfile_individual1)
    assert (
        "PoseEstimation"
        in nwbfile_individual1.processing["behavior"].data_interfaces
    )
    assert (
        "Skeletons"
        in nwbfile_individual1.processing["behavior"].data_interfaces
    )


def test_save_poses_to_multiple_nwb_files():
    ds = sample_data.fetch_dataset("DLC_two-mice.predictions.csv")
    session_start_time = datetime.datetime.now(datetime.UTC)
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
    to_nwb_file(ds, nwbfiles)
