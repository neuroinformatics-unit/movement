import ndx_pose

from movement.io.nwb import _ds_to_pose_and_skeleton_objects


def test_ds_to_pose_and_skeleton_objects(valid_poses_dataset):
    """Test the conversion of a valid poses dataset to
    PoseEstimationSeries and Skeletons objects.
    """
    pose_estimation, skeletons = _ds_to_pose_and_skeleton_objects(
        valid_poses_dataset.sel(individuals="id_0")
    )
    assert isinstance(pose_estimation, list)
    assert isinstance(skeletons, ndx_pose.Skeletons)
    assert (
        set(valid_poses_dataset.keypoints.values)
        == pose_estimation[0].pose_estimation_series.keys()
    )
    assert {"id_0_skeleton"} == skeletons.skeletons.keys()
