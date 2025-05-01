"""Functions to convert between movement poses datasets and NWB files.

The pose tracks in NWB files are formatted according to the ``ndx-pose``
NWB extension, see https://github.com/rly/ndx-pose.
"""

import ndx_pose
import pynwb
import xarray as xr

from movement.utils.logging import logger

DEFAULT_POSE_ESTIMATION_SERIES_KWARGS = dict(
    reference_frame="(0,0,0) corresponds to ...",
)


def _merge_kwargs(defaults, overrides):
    return {**defaults, **(overrides or {})}


def _ds_to_pose_and_skeleton_objects(
    ds: xr.Dataset,
    pose_estimation_series_kwargs: dict | None = None,
    pose_estimation_kwargs: dict | None = None,
    skeleton_kwargs: dict | None = None,
) -> tuple[list[ndx_pose.PoseEstimation], ndx_pose.Skeletons]:
    """Create PoseEstimation and Skeletons objects from a ``movement`` dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        A single-individual ``movement`` poses dataset.
    pose_estimation_series_kwargs : dict, optional
        PoseEstimationSeries keyword arguments.
        See ``ndx_pose``, by default None
    pose_estimation_kwargs : dict, optional
        PoseEstimation keyword arguments. See ``ndx_pose``, by default None
    skeleton_kwargs : dict, optional
        Skeleton keyword arguments. See ``ndx_pose``, by default None

    Returns
    -------
    pose_estimation : list[ndx_pose.PoseEstimation]
        List of PoseEstimation objects
    skeletons : ndx_pose.Skeletons
        Skeletons object containing all skeletons

    """
    # Use default kwargs, but updated with any user-provided kwargs
    pose_estimation_series_kwargs = _merge_kwargs(
        DEFAULT_POSE_ESTIMATION_SERIES_KWARGS, pose_estimation_series_kwargs
    )
    individual = ds.individuals.values.item()
    keypoints = ds.keypoints.values
    pose_estimation_series = []
    for keypoint in keypoints:
        pose_estimation_series.append(
            ndx_pose.PoseEstimationSeries(
                name=keypoint,
                data=ds.sel(keypoints=keypoint).position.values,
                confidence=ds.sel(keypoints=keypoint).confidence.values,
                unit="pixels",
                timestamps=ds.sel(keypoints=keypoint).time.values,
                **pose_estimation_series_kwargs,
            )
        )
    skeleton_list = [
        ndx_pose.Skeleton(
            name=f"{individual}_skeleton",
            nodes=keypoints,
            **(skeleton_kwargs or {}),
        )
    ]
    # Group all PoseEstimationSeries into a PoseEstimation object
    bodyparts_str = ", ".join(keypoints)
    description = (
        f"Estimated positions of {bodyparts_str} for "
        f"{individual} using {ds.source_software}."
    )
    pose_estimation = [
        ndx_pose.PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=pose_estimation_series,
            description=description,
            source_software=ds.source_software,
            skeleton=skeleton_list[-1],
            **(pose_estimation_kwargs or {}),
        )
    ]
    skeletons = ndx_pose.Skeletons(skeletons=skeleton_list)
    return pose_estimation, skeletons


def _write_behavior_processing_module(
    nwb_file: pynwb.NWBFile,
    pose_estimation: ndx_pose.PoseEstimation,
    skeletons: ndx_pose.Skeletons,
) -> None:
    """Write behaviour processing data to an NWB file.

    PoseEstimation or Skeletons objects will be written to the NWB file's
    "behavior" processing module, formatted according to the ``ndx-pose`` NWB
    extension. If the module does not exist, it will be created.
    Data will not overwrite any existing objects in the NWB file.

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
        The NWB file object to which the data will be added.
    pose_estimation : ndx_pose.PoseEstimation
        PoseEstimation object containing the pose data for an individual.
    skeletons : ndx_pose.Skeletons
        Skeletons object containing the skeleton data for an individual.

    """
    try:
        behavior_pm = nwb_file.create_processing_module(
            name="behavior",
            description="processed behavioral data",
        )
        logger.debug("Created behavior processing module in NWB file.")
    except ValueError:
        logger.debug(
            "Data will be added to existing behavior processing module."
        )
        behavior_pm = nwb_file.processing["behavior"]

    try:
        behavior_pm.add(skeletons)
        logger.info("Added Skeletons object to NWB file.")
    except ValueError:
        logger.warning("Skeletons object already exists. Skipping...")

    try:
        behavior_pm.add(pose_estimation)
        logger.info("Added PoseEstimation object to NWB file.")
    except ValueError:
        logger.warning("PoseEstimation object already exists. Skipping...")
