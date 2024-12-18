"""Functions to convert between movement poses datasets and NWB files.

The pose tracks in NWB files are formatted according to the ``ndx-pose``
NWB extension, see https://github.com/rly/ndx-pose.
"""

import ndx_pose
import pynwb
import xarray as xr

from movement.utils.logging import log_error

# Default keyword arguments for Skeletons,
# PoseEstimation and PoseEstimationSeries objects
SKELETON_KWARGS = dict(edges=None)
POSE_ESTIMATION_SERIES_KWARGS = dict(
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
POSE_ESTIMATION_KWARGS = dict(
    original_videos=None,
    labeled_videos=None,
    dimensions=None,
    devices=None,
    scorer=None,
    source_software_version=None,
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
        POSE_ESTIMATION_SERIES_KWARGS, pose_estimation_series_kwargs
    )
    pose_estimation_kwargs = _merge_kwargs(
        POSE_ESTIMATION_KWARGS, pose_estimation_kwargs
    )
    skeleton_kwargs = _merge_kwargs(SKELETON_KWARGS, skeleton_kwargs)

    # Extract individual name
    individual = ds.individuals.values.item()

    # Create a PoseEstimationSeries object for each keypoint
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
    # Create a Skeleton object for the chosen individual
    skeleton_list = [
        ndx_pose.Skeleton(
            name=f"{individual}_skeleton",
            nodes=ds.keypoints.to_numpy().tolist(),
            **skeleton_kwargs,
        )
    ]

    # Group all PoseEstimationSeries into a PoseEstimation object
    bodyparts_str = ", ".join(ds.keypoints.to_numpy().tolist())
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
            **pose_estimation_kwargs,
        )
    ]
    # Create a Skeletons object
    skeletons = ndx_pose.Skeletons(skeletons=skeleton_list)

    return pose_estimation, skeletons


def ds_to_nwb(
    movement_dataset: xr.Dataset,
    nwb_files: pynwb.NWBFile | list[pynwb.NWBFile],
    *,  # Enforce keyword-only arguments henceforth
    pose_estimation_series_kwargs: dict | None = None,
    pose_estimation_kwargs: dict | None = None,
    skeletons_kwargs: dict | None = None,
) -> None:
    """Write a ``movement`` dataset to one or more NWB file objects.

    The data will be written to the NWB file(s) in the "behavior" processing
    module, formatted according to the ``ndx-pose`` NWB extension [1]_.
    Each individual in the dataset will be written to a separate file object,
    as required by the NWB format. Note that the NWBFile object(s) are not
    automatically saved to disk.

    Parameters
    ----------
    movement_dataset : xr.Dataset
        ``movement`` poses dataset containing the data to be converted to NWB
    nwb_files : list[pynwb.NWBFile] | pynwb.NWBFile
        An NWBFile object or a list of such objects to which the data
        will be added.
    pose_estimation_series_kwargs : dict, optional
        PoseEstimationSeries keyword arguments.
        See ``ndx-pose``, by default None
    pose_estimation_kwargs : dict, optional
        PoseEstimation keyword arguments. See ``ndx-pose``, by default None
    skeletons_kwargs : dict, optional
        Skeleton keyword arguments. See ``ndx-pose``, by default None

    Raises
    ------
    ValueError
        If the number of NWBFiles is not equal to the number of individuals
        in the dataset.

    References
    ----------
    .. [1] https://github.com/rly/ndx-pose

    """
    if isinstance(nwb_files, pynwb.NWBFile):
        nwb_files = [nwb_files]

    if len(nwb_files) != len(movement_dataset.individuals):
        raise log_error(
            ValueError,
            "Number of NWBFile objects must be equal to the number of "
            "individuals, as NWB requires one file per individual (subject).",
        )

    for nwb_file, individual in zip(
        nwb_files, movement_dataset.individuals.values, strict=False
    ):
        pose_estimation, skeletons = _ds_to_pose_and_skeleton_objects(
            movement_dataset.sel(individuals=individual),
            pose_estimation_series_kwargs,
            pose_estimation_kwargs,
            skeletons_kwargs,
        )
        try:
            behavior_pm = nwb_file.create_processing_module(
                name="behavior",
                description="processed behavioral data",
            )
        except ValueError:
            print("Behavior processing module already exists. Skipping...")
            behavior_pm = nwb_file.processing["behavior"]

        try:
            behavior_pm.add(skeletons)
        except ValueError:
            print("Skeletons already exists. Skipping...")
        try:
            behavior_pm.add(pose_estimation)
        except ValueError:
            print("PoseEstimation already exists. Skipping...")
