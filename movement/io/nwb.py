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
    unit="pixels",
)
ConfigKwargsType = dict[str, Any] | dict[str, dict[str, Any]]


def _safe_dict_field() -> Any:
    """Create a field that defaults to an empty dictionary."""
    return field(
        factory=dict, converter=lambda x: x if isinstance(x, dict) else {}
    )


@define(kw_only=True)
class NWBFileSaveConfig:
    """Configuration for saving ``movement poses`` dataset to NWBFile(s).

    This class is used with :func:`movement.io.save_poses.to_nwb_file`
    to add custom metadata to the NWBFile(s) created from a given
    ``movement`` dataset.
    TODO: Clean up docstrings, might need to move some parts to
    to_nwb_file

    Attributes
    ----------
    nwbfile_kwargs : dict[str, Any] | dict[str, dict[str, any]], optional
        Keyword arguments for :class:`pynwb.file.NWBFile`.

        If ``nwbfile_kwargs`` is a single dictionary, the same keyword
        arguments will be applied to all NWBFile objects except for
        ``identifier``.

        If ``nwbfile_kwargs`` is a dictionary of dictionaries, the outer keys
        should correspond to individual names in the ``movement`` dataset,
        and the inner dictionaries will be passed as keyword arguments to the
        :class:`pynwb.file.NWBFile` constructor.

        The following arguments will have default values:

        - ``session_description``: "not set"
        - ``session_start_time``: current UTC time
        - ``identifier``: "not set"

        If specified, ``identifier`` will be set in the following
        order of precedence:

        1. ``identifier`` in the inner dictionary
        2. ``nwbfile_kwargs["identifier"]`` (single-individual dataset only)
        3. individual name in the ``movement`` dataset

    subject_kwargs : dict[str, Any] | dict[str, dict[str, any]], optional
        Keyword arguments for :class:`pynwb.file.Subject`.

        If ``subject_kwargs`` is a single dictionary, the same keyword
        arguments will be applied to all Subjects except for ``subject_id``.

        If ``subject_kwargs`` is a dictionary of dictionaries, the outer keys
        should correspond to individual names in the ``movement`` dataset,
        and the inner dictionaries will be passed as keyword arguments to the
        :class:`pynwb.file.Subject` constructor.

        If specified, ``subject_id`` will be set in the following
        order of precedence:

        1. ``subject_id`` in the inner dictionary
        2. ``subject_kwargs["subject_id"]`` (single-individual dataset only)
        3. individual name in the ``movement`` dataset

    pose_estimation_series_kwargs :
            dict[str, Any] | dict[str, dict[str, any]], optional
        Keyword arguments for ``ndx_pose.PoseEstimationSeries`` [1]_.

        If ``pose_estimation_series_kwargs`` is a single dictionary, the same
        keyword arguments will be applied to all PoseEstimationSeries objects.

        If ``pose_estimation_series_kwargs`` is a dictionary of dictionaries,
        the outer keys should correspond to keypoint names in the
        ``movement`` dataset, and the inner dictionaries will be passed as
        keyword arguments to the ``ndx_pose.PoseEstimationSeries`` constructor.

        The following arguments will be set based on the dataset:

        - ``data``: position data for the keypoint
        - ``confidence``: confidence data for the keypoint
        - ``timestamps``: time data for the keypoint

        The following arguments will have default values:

        - ``unit``: "pixels"
        - ``reference_frame``: "(0,0,0) corresponds to ..."

        If specified, ``name`` will be set in the following
        order of precedence:

        1. ``name`` in the inner dictionary
        2. ``pose_estimation_series_kwargs["name"]`` (single-keypoint
           dataset only)
        3. keypoint name in the ``movement`` dataset

    References
    ----------
    .. [1] https://github.com/rly/ndx-pose

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

    nwbfile_kwargs: ConfigKwargsType = _safe_dict_field()
    subject_kwargs: ConfigKwargsType = _safe_dict_field()
    pose_estimation_series_kwargs: ConfigKwargsType = _safe_dict_field()
    pose_estimation_kwargs: ConfigKwargsType = _safe_dict_field()
    skeletons_kwargs: ConfigKwargsType = _safe_dict_field()
    DEFAULT_POSE_ESTIMATION_SERIES_KWARGS = dict(
        reference_frame="(0,0,0) corresponds to ...", unit="pixels"
    )
    DEFAULT_NWBFILE_KWARGS = dict(
        session_description="not set", identifier="not set"
    )

    def _resolve_kwargs(
        self,
        attr_name: str,
        entity: str | None,
        entity_type: str,
        id_key: str,
        prioritise_entity: bool = True,
    ) -> dict[str, Any]:
        """Resolve per-entity (individual/keypoint) or shared kwargs.

        If the kwargs attribute (retrieved from ``attr_name``) is a
        dictionary of dictionaries, the outer keys should correspond to
        individual or keypoint names in the ``movement`` dataset,
        and the inner dictionaries will be returned as keyword arguments
        for an individual/keypoint.

        If the retrieved attribute is a single dictionary, the same arguments
        will be used for all individuals or keypoints except for ``id_key``,
        which will be set in the following order of precedence:
        1. ``entity`` (if provided)
        2. ``kwargs[id_key]`` (if provided)
        3. ``DEFAULT_<attr_name>[id_key]`` (class attribute)

        Parameters
        ----------
        attr_name : str
            The name of the attribute in the class (e.g. ``nwbfile_kwargs``,
            ``subject_kwargs``, ``pose_estimation_series_kwargs``) to be
            resolved.
        entity : str | None
            Individual or keypoint name.
        entity_type : str | None
            Type of entity (i.e. "individual", "keypoint"). Used in error
            or warning messages.
        id_key : str
            The key in ``cfg`` corresponding to the entity identifier/name
            (e.g. ``identifier``, ``subject_id``, ``name``) to be set in the
            returned dictionary.
        prioritise_entity : bool, optional
            Flag indicating whether ``entity`` should take
            precedence over the ``id_key`` when the attribute is a shared
            config (i.e. a single dictionary). Default is True.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for a specific ``entity_type``.

        """

        def infer_entity_or_raise(cfg: dict) -> str:
            if len(cfg) == 1:
                inferred = next(iter(cfg))
                logger.warning(
                    f"No {entity_type} was provided. Assuming '{inferred}' "
                    f"since there is only one entry in {attr_name}."
                )
                return inferred
            raise logger.error(
                ValueError(
                    f"NWBFileSaveConfig has per-{entity_type} "
                    f"{attr_name}, but no {entity_type} was provided."
                )
            )

        cfg = getattr(self, attr_name)
        defaults = getattr(self, f"DEFAULT_{attr_name.upper()}", None)

        if self._is_per_entity_config(cfg):
            if entity is None:
                entity = infer_entity_or_raise(cfg)
            base = dict(cfg.get(entity, {}))
            if not base:
                logger.warning(
                    f"'{entity}' not found in {attr_name}; "
                    f"setting only {id_key} to '{entity}'."
                )
            base.setdefault(id_key, entity)
        else:  # Shared config
            base = dict(cfg)
            if not prioritise_entity:
                base[id_key] = cfg.get(id_key, entity)
            elif entity is not None:
                base[id_key] = entity
        if defaults:
            base = {**defaults, **base}  # base overrides defaults
        return base

    def resolve_nwbfile_kwargs(
        self, individual: str | None = None, prioritise_individual: bool = True
    ) -> dict[str, Any]:
        """Resolve the keyword arguments for :class:`pynwb.file.NWBFile`.

        Parameters
        ----------
        individual : str, optional
            Individual name. If provided, the method will attempt to retrieve
            individual-specific settings or fall back to shared or default
            settings.
        prioritise_individual: bool, optional
            Flag indicating whether ``individual`` should take
            precedence over the ``identifier`` in shared
            ``nwbfile_kwargs``.
            Default is True.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to be passed to :class:`pynwb.file.NWBFile`.

        """
        kwargs = self._resolve_kwargs(
            attr_name="nwbfile_kwargs",
            entity=individual,
            entity_type="individual",
            id_key="identifier",
            prioritise_entity=prioritise_individual,
        )
        kwargs.setdefault(
            "session_start_time", datetime.datetime.now(datetime.UTC)
        )
        return kwargs

    def resolve_subject_kwargs(
        self, individual: str | None = None, prioritise_individual: bool = True
    ) -> dict[str, Any]:
        """Resolve the keyword arguments for :class:`pynwb.file.Subject`.

        Parameters
        ----------
        individual : str, optional
            Individual name. If provided, the method will attempt to retrieve
            individual-specific settings or fall back to shared or default
            settings.
        prioritise_individual: bool, optional
            Flag indicating whether ``individual`` should take
            precedence over the ``subject_id`` in shared
            ``subject_kwargs``.
            Default is True.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to be passed to :class:`pynwb.file.Subject`.

        """
        return self._resolve_kwargs(
            attr_name="subject_kwargs",
            entity=individual,
            entity_type="individual",
            id_key="subject_id",
            prioritise_entity=prioritise_individual,
        )

    def resolve_pose_estimation_series_kwargs(
        self, keypoint: str | None = None, prioritise_keypoint: bool = True
    ) -> dict[str, Any]:
        """Resolve the keyword arguments for ``ndx_pose.PoseEstimationSeries``.

        Parameters
        ----------
        keypoint : str, optional
            Keypoint name. If provided, the method will attempt to retrieve
            keypoint-specific settings or fall back to shared or default
            settings.
        prioritise_keypoint: bool, optional
            Flag indicating whether ``keypoint`` should take
            precedence over the ``name`` in shared
            ``pose_estimation_series_kwargs``.
            Default is True.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to be passed to
            ``ndx_pose.PoseEstimationSeries``.

        """
        return self._resolve_kwargs(
            attr_name="pose_estimation_series_kwargs",
            entity=keypoint,
            entity_type="keypoint",
            id_key="name",
            prioritise_entity=prioritise_keypoint,
        )

    def _is_per_entity_config(self, cfg: dict) -> bool:
        return bool(cfg) and all(isinstance(v, dict) for v in cfg.values())


def _merge_kwargs(defaults, overrides):
    return {**defaults, **(overrides or {})}


def _ds_to_pose_and_skeletons(
    ds: xr.Dataset,
    config: NWBFileSaveConfig | None = None,
    subject: pynwb.file.Subject | None = None,
) -> tuple[ndx_pose.PoseEstimation, ndx_pose.Skeletons]:
    """Create PoseEstimation and Skeletons objects from a ``movement`` dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        A single-individual ``movement`` poses dataset.
    config : movement.io.NWBFileSaveConfig
        Configuration object containing keyword arguments to customise
        the PoseEstimation and Skeletons objects created from the dataset.
        If None (default), default values will be used.
        See :class:`movement.io.NWBFileSaveConfig` for more details.
    subject : pynwb.file.Subject, optional
        Subject object to be linked in the Skeleton object.

    Returns
    -------
    pose_estimation : ndx_pose.PoseEstimation
        PoseEstimation object containing PoseEstimationSeries objects
        for each keypoint in the dataset.
    skeletons : ndx_pose.Skeletons
        Skeletons object containing all Skeleton objects.

    """
    if ds.individuals.size != 1:
        raise logger.error(
            ValueError(
                "Dataset must contain only one individual to create "
                "PoseEstimation and Skeletons objects."
            )
        )
    config = config or NWBFileSaveConfig()
    skeleton_kwargs = dict(config.skeletons_kwargs)
    individual = ds.individuals.values.item()
    keypoints = ds.keypoints.values.tolist()
    pose_estimation_series = [
        ndx_pose.PoseEstimationSeries(
            data=ds.sel(keypoints=keypoint).position.values,
            confidence=ds.sel(keypoints=keypoint).confidence.values,
            timestamps=ds.time.values,
            **(
                config.resolve_pose_estimation_series_kwargs(
                    keypoint, len(keypoints) > 1
                )
            ),
        )
        for keypoint in keypoints
    ]
    skeleton_list = [
        ndx_pose.Skeleton(
            name=f"{individual}_skeleton",
            nodes=skeleton_kwargs.pop("nodes", keypoints),
            subject=subject,
            **skeleton_kwargs,
        )
    ]
    # Group all PoseEstimationSeries into a PoseEstimation object
    bodyparts_str = ", ".join(keypoints)
    description = (
        f"Estimated positions of {bodyparts_str} for "
        f"{individual} using {ds.source_software}."
    )
    pose_estimation = ndx_pose.PoseEstimation(
        name="PoseEstimation",
        pose_estimation_series=pose_estimation_series,
        description=description,
        source_software=ds.source_software,
        skeleton=skeleton_list[-1],
        **config.pose_estimation_kwargs,
    )
    skeletons = ndx_pose.Skeletons(skeletons=skeleton_list)
    return pose_estimation, skeletons


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
