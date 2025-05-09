"""Functions to convert between movement poses datasets and NWB files.

The pose tracks in NWB files are formatted according to the ``ndx-pose``
NWB extension, see https://github.com/rly/ndx-pose.
"""

import datetime
from typing import Any

import ndx_pose
import pynwb
import xarray as xr
from attrs import define, field

from movement.utils.logging import logger

DEFAULT_POSE_ESTIMATION_SERIES_KWARGS = dict(
    reference_frame="(0,0,0) corresponds to ...",
)
ConfigKwargsType = dict[str, Any] | dict[str, dict[str, Any]]


@define(kw_only=True)
class NWBFileSaveConfig:
    """Configuration for saving ``movement poses`` dataset to NWBFile(s).

    All fields are optional and will default to empty dictionaries.

    Attributes
    ----------
    nwbfile_kwargs : dict[str, Any] | dict[str, dict[str, any]], optional
        Keyword arguments for the :class:`pynwb.file.NWBFile` constructor.
        If ``nwbfile_kwargs`` is a dictionary of dictionaries, the outer keys
        should correspond to individual names in the ``movement`` dataset,
        and the inner dictionaries will be passed as keyword arguments to the
        :class:`pynwb.file.NWBFile` constructor. If ``nwbfile_kwargs`` is a
        single dictionary, the same arguments will be applied to all NWBFile
        objects except for ``identifier``, which will be set to the individual
        name.
    subject_kwargs : dict[str, Any] | dict[str, dict[str, any]], optional
        Keyword arguments for the :class:`pynwb.file.Subject` constructor.
        If ``subject_kwargs`` is a dictionary of dictionaries, the outer keys
        should correspond to individual names in the ``movement`` dataset,
        and the inner dictionaries will be passed as keyword arguments to the
        :class:`pynwb.file.Subject` constructor. If ``subject_kwargs`` is a
        single dictionary, the same arguments will be applied to all Subjects
        except for ``subject_id``, which will be set to the individual name.

    Examples
    --------
    Create :class:`pynwb.file.NWBFile` objects with the same ``nwbfile_kwargs``
    and the same ``subject_kwargs`` for all individuals (e.g. `id_0`, `id_1`)
    in the dataset:

    >>> from movement.io.nwb import NWBFileSaveConfig
    >>> from movement.io.save_poses import to_nwb_file
    >>> config = NWBFileSaveConfig(
    ...     nwbfile_kwargs={"session_description": "test session"},
    ...     subject_kwargs={"age": "1 year", "species": "mouse"},
    ... )
    >>> nwb_files = to_nwb_file(ds, config)

    Create :class:`pynwb.file.NWBFile` objects with different
    ``nwbfile_kwargs`` and ``subject_kwargs`` for each individual
    (e.g. `id_0`, `id_1`) in the dataset:

    >>> config = NWBFileSaveConfig(
    ...     nwbfile_kwargs={
    ...         "id_0": {
    ...             "identifier": "subj1",
    ...             "session_description": "subj1 session",
    ...         },
    ...         "id_1": {
    ...             "identifier": "subj2",
    ...             "session_description": "subj2 session",
    ...         },
    ...     },
    ...     subject_kwargs={
    ...         "id_0": {"age": "1 year", "species": "mouse"},
    ...         "id_1": {"age": "2 years", "species": "rat"},
    ...     },
    ...     pose_estimation_series_kwargs={"reference_frame": "(0,0,0)"},
    ...     pose_estimation_kwargs={"description": "test description"},
    ...     skeletons_kwargs={"nodes": ["nose", "left_eye", "right_eye"]},
    ... )
    >>> nwb_files = to_nwb_file(ds, config)

    """

    nwbfile_kwargs: ConfigKwargsType = field(
        factory=dict, converter=lambda x: x if isinstance(x, dict) else {}
    )
    subject_kwargs: ConfigKwargsType = field(
        factory=dict, converter=lambda x: x if isinstance(x, dict) else {}
    )
    pose_estimation_series_kwargs: ConfigKwargsType = field(
        factory=dict, converter=lambda x: x if isinstance(x, dict) else {}
    )
    pose_estimation_kwargs: ConfigKwargsType = field(
        factory=dict, converter=lambda x: x if isinstance(x, dict) else {}
    )
    skeletons_kwargs: ConfigKwargsType = field(
        factory=dict, converter=lambda x: x if isinstance(x, dict) else {}
    )

    def _resolve_kwargs(
        self,
        cfg: ConfigKwargsType,
        individual: str | None,
        id_key: str,
        warn_context: str,
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve per-individual or shared kwargs.

        If ``cfg`` is a dictionary of dictionaries, the outer keys
        should correspond to individual names in the ``movement`` dataset,
        and the inner dictionaries will be returned as keyword arguments
        for an individual.

        If ``cfg`` is a single dictionary, the same arguments will
        be used for all individuals, except for ``id_key``,
        which will be set in the following order of precedence:
        1. ``individual`` (if provided)
        2. ``cfg[id_key]`` (if provided)
        3. ``defaults[id_key]`` (if provided)
        """
        is_per_individual = (
            cfg
            and isinstance(cfg, dict)
            and all(isinstance(v, dict) for v in cfg.values())
        )
        if is_per_individual:
            if individual is None:
                if len(cfg) == 1:
                    individual = next(iter(cfg.keys()))
                    logger.warning(
                        f"No individual was provided. Assuming '{individual}' "
                        f"as the individual since there is only one "
                        f"configuration entry in {warn_context}."
                    )
                else:
                    raise logger.error(
                        ValueError(
                            "NWBFileSaveConfig has per-individual "
                            f"{warn_context}, but no individual was provided."
                        )
                    )
            if individual in cfg:
                base = dict(cfg[individual])
                base.setdefault(id_key, individual)
            else:
                logger.warning(
                    f"Individual '{individual}' not found in {warn_context}; "
                    f"setting only {id_key} to '{individual}'."
                )
                base = {id_key: individual}
        else:
            base = dict(cfg)
            if individual is not None:
                base[id_key] = individual

        if defaults:
            for k, v in defaults.items():
                base.setdefault(k, v)
        return base

    def resolve_nwbfile_kwargs(
        self, individual: str | None = None
    ) -> dict[str, Any]:
        """Resolve the keyword arguments for :class:`pynwb.file.NWBFile`.

        If ``nwbfile_kwargs`` is a dictionary of dictionaries, the outer keys
        should correspond to individual names in the ``movement`` dataset,
        and the inner dictionaries will be passed as keyword arguments to the
        :class:`pynwb.file.NWBFile` constructor.

        If ``nwbfile_kwargs`` is a single dictionary, the same arguments will
        be applied to all NWBFile objects, except for ``identifier``,
        which will be set in the following order of precedence:

        1. ``individual`` (if provided)
        2. ``nwbfile_kwargs["identifier"]`` (if provided)
        3. default value ``"not set"``

        Parameters
        ----------
        individual : str, optional
            Individual name. If provided, the method will attempt to retrieve
            individual-specific settings or fall back to shared or default
            settings.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to be passed to :class:`pynwb.file.NWBFile`.

        """
        defaults = {
            "session_description": "not set",
            "session_start_time": datetime.datetime.now(datetime.UTC),
            "identifier": "not set",
        }
        return self._resolve_kwargs(
            cfg=self.nwbfile_kwargs,
            individual=individual,
            id_key="identifier",
            warn_context="nwbfile_kwargs",
            defaults=defaults,
        )

    def resolve_subject_kwargs(
        self, individual: str | None = None
    ) -> dict[str, Any]:
        """Resolve the keyword arguments for :class:`pynwb.file.Subject`.

        If ``subject_kwargs`` is a dictionary of dictionaries, the outer keys
        should correspond to individual names in the ``movement`` dataset,
        and the inner dictionaries will be passed as keyword arguments to the
        :class:`pynwb.file.Subject` constructor.

        If ``subject_kwargs`` is a single dictionary, the same arguments will
        be applied to all Subject objects, except for ``subject_id``,
        which will be set in the following order of precedence:
        1. ``individual`` (if provided)
        2. ``subject_kwargs["subject_id"]`` (if provided)

        Parameters
        ----------
        individual : str, optional
            Individual name. If provided, the method will attempt to retrieve
            individual-specific settings or fall back to shared or default
            settings.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to be passed to :class:`pynwb.file.Subject`.

        """
        return self._resolve_kwargs(
            cfg=self.subject_kwargs,
            individual=individual,
            id_key="subject_id",
            warn_context="subject_kwargs",
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
    skeleton_kwargs = skeleton_kwargs or {}
    individual = ds.individuals.values.item()
    keypoints = ds.keypoints.values.tolist()
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
            nodes=skeleton_kwargs.pop("nodes", keypoints),
            **skeleton_kwargs,
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
