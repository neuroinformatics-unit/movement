from typing import Optional, Union

import ndx_pose
import numpy as np
import pynwb
import xarray as xr


def _create_pose_and_skeleton_objects(
    ds: xr.Dataset,
    subject: str,
    pose_estimation_series_kwargs: Optional[dict] = None,
    pose_estimation_kwargs: Optional[dict] = None,
    skeleton_kwargs: Optional[dict] = None,
) -> tuple[list[ndx_pose.PoseEstimation], ndx_pose.Skeletons]:
    """Creates PoseEstimation and Skeletons objects from a movement xarray
    dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset containing the data to be converted to NWB.
    subject : str
        Name of the subject to be converted.
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

    pose_estimation_series = []

    for keypoint in ds.keypoints.to_numpy():
        pose_estimation_series.append(
            ndx_pose.PoseEstimationSeries(
                name=keypoint,
                data=ds.sel(keypoints=keypoint).position.to_numpy(),
                confidence=ds.sel(keypoints=keypoint).confidence.to_numpy(),
                unit="pixels",
                timestamps=ds.sel(keypoints=keypoint).time.to_numpy(),
                **pose_estimation_series_kwargs,
            )
        )

    skeleton_list = [
        ndx_pose.Skeleton(
            name=f"{subject}_skeleton",
            nodes=ds.keypoints.to_numpy().tolist(),
            **skeleton_kwargs,
        )
    ]

    bodyparts_str = ", ".join(ds.keypoints.to_numpy().tolist())
    description = (
        f"Estimated positions of {bodyparts_str} of"
        f"{subject} using {ds.source_software}."
    )

    pose_estimation = [
        ndx_pose.PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=pose_estimation_series,
            description=description,
            source_software=ds.source_software,
            skeleton=skeleton_list[-1],
            **pose_estimation_kwargs,
        )
    ]

    skeletons = ndx_pose.Skeletons(skeletons=skeleton_list)

    return pose_estimation, skeletons


def convert_movement_to_nwb(
    nwbfiles: Union[list[pynwb.NWBFile], pynwb.NWBFile],
    ds: xr.Dataset,
    pose_estimation_series_kwargs: Optional[dict] = None,
    pose_estimation_kwargs: Optional[dict] = None,
    skeletons_kwargs: Optional[dict] = None,
):
    if isinstance(nwbfiles, pynwb.NWBFile):
        nwbfiles = [nwbfiles]

    if len(nwbfiles) != len(ds.individuals):
        raise ValueError(
            "Number of NWBFiles must be equal to the number of individuals. "
            "NWB requires one file per individual."
        )

    for nwbfile, subject in zip(nwbfiles, ds.individuals.to_numpy()):
        pose_estimation, skeletons = _create_pose_and_skeleton_objects(
            ds.sel(individuals=subject),
            subject,
            pose_estimation_series_kwargs,
            pose_estimation_kwargs,
            skeletons_kwargs,
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


def _convert_pse(
    pes: ndx_pose.PoseEstimationSeries,
    keypoint: str,
    subject_name: str,
    source_software: str,
    source_file: Optional[str] = None,
):
    attrs = {
        "fps": int(np.median(1 / np.diff(pes.timestamps))),
        "time_units": pes.timestamps_unit,
        "source_software": source_software,
        "source_file": source_file,
    }
    n_space_dims = pes.data.shape[1]
    space_dims = ["x", "y", "z"]

    return xr.Dataset(
        data_vars={
            "position": (
                ["time", "individuals", "keypoints", "space"],
                pes.data[:, np.newaxis, np.newaxis, :],
            ),
            "confidence": (
                ["time", "individuals", "keypoints"],
                pes.confidence[:, np.newaxis, np.newaxis],
            ),
        },
        coords={
            "time": pes.timestamps,
            "individuals": [subject_name],
            "keypoints": [keypoint],
            "space": space_dims[:n_space_dims],
        },
        attrs=attrs,
    )


def convert_nwb_to_movement(nwb_filepaths: list[str]) -> xr.Dataset:
    datasets = []
    for path in nwb_filepaths:
        with pynwb.NWBHDF5IO(path, mode="r") as io:
            nwbfile = io.read()
            pose_estimation = nwbfile.processing["behavior"]["PoseEstimation"]
            source_software = pose_estimation.fields["source_software"]
            pose_estimation_series = pose_estimation.fields[
                "pose_estimation_series"
            ]

            for keypoint, pes in pose_estimation_series.items():
                datasets.append(
                    _convert_pse(
                        pes,
                        keypoint,
                        subject_name=nwbfile.identifier,
                        source_software=source_software,
                        source_file=None,
                    )
                )

    return xr.merge(datasets)
