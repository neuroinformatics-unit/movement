"""Functions to convert movement data to and from NWB format."""

from pathlib import Path

import ndx_pose
import numpy as np
import pynwb
import xarray as xr

from movement.io.save_poses import _validate_file_path
from movement.logging import log_error


def _create_pose_and_skeleton_objects(
    ds: xr.Dataset,
    subject: str,
    pose_estimation_series_kwargs: dict | None = None,
    pose_estimation_kwargs: dict | None = None,
    skeleton_kwargs: dict | None = None,
) -> tuple[list[ndx_pose.PoseEstimation], ndx_pose.Skeletons]:
    """Create PoseEstimation and Skeletons objects from a ``movement`` dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        movement dataset containing the data to be converted to NWB.
    subject : str
        Name of the subject (individual) to be converted.
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


def add_movement_dataset_to_nwb(
    nwbfiles: list[pynwb.NWBFile] | pynwb.NWBFile,
    movement_dataset: xr.Dataset,
    pose_estimation_series_kwargs: dict | None = None,
    pose_estimation_kwargs: dict | None = None,
    skeletons_kwargs: dict | None = None,
) -> None:
    """Add pose estimation data to NWB files for each individual.

    Parameters
    ----------
    nwbfiles : list[pynwb.NWBFile] | pynwb.NWBFile
        NWBFile object(s) to which the data will be added.
    movement_dataset : xr.Dataset
        ``movement`` dataset containing the data to be converted to NWB.
    pose_estimation_series_kwargs : dict, optional
        PoseEstimationSeries keyword arguments. See ndx_pose, by default None
    pose_estimation_kwargs : dict, optional
        PoseEstimation keyword arguments. See ndx_pose, by default None
    skeletons_kwargs : dict, optional
        Skeleton keyword arguments. See ndx_pose, by default None

    Raises
    ------
    ValueError
        If the number of NWBFiles is not equal to the number of individuals
        in the dataset.

    """
    if isinstance(nwbfiles, pynwb.NWBFile):
        nwbfiles = [nwbfiles]

    if len(nwbfiles) != len(movement_dataset.individuals):
        raise log_error(
            ValueError,
            "Number of NWBFiles must be equal to the number of individuals. "
            "NWB requires one file per individual.",
        )

    for nwbfile, subject in zip(
        nwbfiles, movement_dataset.individuals.to_numpy(), strict=False
    ):
        pose_estimation, skeletons = _create_pose_and_skeleton_objects(
            movement_dataset.sel(individuals=subject),
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


def _convert_pose_estimation_series(
    pose_estimation_series: ndx_pose.PoseEstimationSeries,
    keypoint: str,
    subject_name: str,
    source_software: str,
    source_file: str | None = None,
) -> xr.Dataset:
    """Convert to single-keypoint, single-individual ``movement`` dataset.

    Parameters
    ----------
    pose_estimation_series : ndx_pose.PoseEstimationSeries
        PoseEstimationSeries NWB object to be converted.
    keypoint : str
        Name of the keypoint - body part.
    subject_name : str
        Name of the subject (individual).
    source_software : str
        Name of the software used to estimate the pose.
    source_file : Optional[str], optional
        File from which the data was extracted, by default None

    Returns
    -------
    movement_dataset : xr.Dataset
        ``movement`` compatible dataset containing the pose estimation data.

    """
    attrs = {
        "fps": np.nanmedian(1 / np.diff(pose_estimation_series.timestamps)),
        "time_units": pose_estimation_series.timestamps_unit,
        "source_software": source_software,
        "source_file": source_file,
    }
    n_space_dims = pose_estimation_series.data.shape[1]
    space_dims = ["x", "y", "z"]

    position_array = np.asarray(pose_estimation_series.data)[
        :, np.newaxis, np.newaxis, :
    ]

    if getattr(pose_estimation_series, "confidence", None) is None:
        pose_estimation_series.confidence = np.full(
            pose_estimation_series.data.shape[0], np.nan
        )
    else:
        confidence_array = np.asarray(pose_estimation_series.confidence)[
            :, np.newaxis, np.newaxis
        ]

    return xr.Dataset(
        data_vars={
            "position": (
                ["time", "individuals", "keypoints", "space"],
                position_array,
            ),
            "confidence": (
                ["time", "individuals", "keypoints"],
                confidence_array,
            ),
        },
        coords={
            "time": pose_estimation_series.timestamps,
            "individuals": [subject_name],
            "keypoints": [keypoint],
            "space": space_dims[:n_space_dims],
        },
        attrs=attrs,
    )


def convert_nwb_to_movement(
    nwb_filepaths: str | list[str] | list[Path],
) -> xr.Dataset:
    """Convert a list of NWB files to a single ``movement`` dataset.

    Parameters
    ----------
    nwb_filepaths : str | Path | list[str] | list[Path]
        List of paths to NWB files to be converted.

    Returns
    -------
    movement_ds : xr.Dataset
        ``movement`` dataset containing the pose estimation data.

    """
    if isinstance(nwb_filepaths, str | Path):
        nwb_filepaths = [nwb_filepaths]

    datasets = []
    for path in nwb_filepaths:
        _validate_file_path(path, expected_suffix=[".nwb"])
        with pynwb.NWBHDF5IO(path, mode="r") as io:
            nwbfile = io.read()
            pose_estimation = nwbfile.processing["behavior"]["PoseEstimation"]
            source_software = pose_estimation.fields["source_software"]
            pose_estimation_series = pose_estimation.fields[
                "pose_estimation_series"
            ]

            for keypoint, pes in pose_estimation_series.items():
                datasets.append(
                    _convert_pose_estimation_series(
                        pes,
                        keypoint,
                        subject_name=nwbfile.identifier,
                        source_software=source_software,
                        source_file=None,
                    )
                )

    return xr.merge(datasets)
