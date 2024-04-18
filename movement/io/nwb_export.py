import xarray as xr

try:
    import ndx_pose
    import pynwb
except ImportError:
    ndx_pose = None
    pynwb = None


def _create_pose_and_skeleton_objects(
    ds: xr.Dataset,
    pose_estimation_series_kwargs: dict = None,
    pose_estimation_kwargs: dict = None,
    skeleton_kwargs: dict = None,
) -> tuple[list[ndx_pose.PoseEstimation], ndx_pose.Skeletons]:
    """Creates PoseEstimation and Skeletons objects from a movement xarray dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset containing the data to be converted to NWB.
    pose_estimation_series_kwargs : dict, optional
        PoseEstimationSeries keyword arguments. See ndx_pose, by default None
    pose_estimation_kwargs : dict, optional
        PoseEstimation keyword arguments. See ndx_pose, by default None
    skeleton_kwargs : dict, optional
        Skeleton keyword arguments. See ndx_pose, by default None

    Returns
    -------
    pose_estimation : list[ndx_pose.PoseEstimation]
        List of PoseEstimation objects
    skeletons : ndx_pose.Skeletons
        Skeletons object containing all skeletons
    """
    if pose_estimation_series_kwargs is None:
        pose_estimation_series_kwargs = dict(
            reference_frame="(0,0,0) corresponds to ...",
            confidence_definition=None,
            conversion=1.0,
            resolution=-1.0,
            offset=0.0,
            starting_time=None,
            comments="no comments",
            description="no description",
            control=None,
            control_description=None,
        )

    if skeleton_kwargs is None:
        skeleton_kwargs = dict(edges=None)

    if pose_estimation_kwargs is None:
        pose_estimation_kwargs = dict(
            original_videos=None,
            labeled_videos=None,
            dimensions=None,
            devices=None,
            scorer=None,
            source_software_version=None,
        )

    skeleton_list = []
    pose_estimation = []

    for subject in ds.individuals.to_numpy():
        pose_estimation_series = []

        for keypoint in ds.keypoints.to_numpy():
            pose_estimation_series.append(
                ndx_pose.PoseEstimationSeries(
                    name=keypoint,
                    data=ds.sel(
                        keypoints=keypoint, individuals=subject
                    ).position.to_numpy(),
                    confidence=ds.sel(
                        keypoints=keypoint, individuals=subject
                    ).confidence.to_numpy(),
                    unit="pixels",
                    timestamps=ds.sel(
                        keypoints=keypoint, individuals=subject
                    ).time.to_numpy(),
                    **pose_estimation_series_kwargs,
                )
            )

        skeleton_list.append(
            ndx_pose.Skeleton(
                name=f"{subject}_skeleton",
                nodes=ds.sel(individuals=subject).keypoints.to_numpy().tolist(),
                **skeleton_kwargs,
            )
        )

        bodyparts_str = ", ".join(
            ds.sel(individuals=subject).keypoints.to_numpy().tolist()
        )
        description = f"Estimated positions of {bodyparts_str} of {subject} using {ds.source_software}."

        pose_estimation.append(
            ndx_pose.PoseEstimation(
                name="PoseEstimation",
                pose_estimation_series=pose_estimation_series,
                description=description,
                source_software=ds.source_software,
                skeleton=skeleton_list[-1],
                **pose_estimation_kwargs,
            )
        )

    skeletons = ndx_pose.Skeletons(skeletons=skeleton_list)

    return pose_estimation, skeletons


def convert_movement_to_nwb(
    nwbfile: pynwb.NWBFile,
    ds: xr.Dataset,
    pose_estimation_series_kwargs: dict = None,
    pose_estimation_kwargs: dict = None,
    skeletons_kwargs: dict = None,
):
    pose_estimation, skeletons = _create_pose_and_skeleton_objects(
        ds, pose_estimation_series_kwargs, pose_estimation_kwargs, skeletons_kwargs
    )
    try:
        behavior_pm = nwbfile.create_processing_module(
            name="behavior",
            description="processed behavioral data",
        )
    except ValueError:
        print("Behavior processing module already exists. Skipping...")
        behavior_pm = nwbfile.processing["behavior"]

    try:
        behavior_pm.add(skeletons)
    except ValueError:
        print("Skeletons already exists. Skipping...")
    try:
        behavior_pm.add(pose_estimation)
    except ValueError:
        print("PoseEstimation already exists. Skipping...")
