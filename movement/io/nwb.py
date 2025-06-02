"""Helpers to convert between movement poses datasets and NWB files.

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

    Attributes
    ----------
    nwbfile_kwargs : dict[str, Any] or dict[str, dict[str, Any]], optional
        Keyword arguments for :class:`pynwb.file.NWBFile`.

        If ``nwbfile_kwargs`` is a single dictionary, the same keyword
        arguments will be applied to all NWBFile objects except for
        ``identifier``.

        If ``nwbfile_kwargs`` is a dictionary of dictionaries, the outer keys
        should correspond to individual names in the ``movement`` dataset,
        and the inner dictionaries will be passed as keyword arguments to the
        :class:`pynwb.file.NWBFile` constructor.

        The following arguments cannot be overwritten:

        - ``subject``: :class:`pynwb.file.Subject` created for the individual
          using ``subject_kwargs``

        The following arguments will have default values if not set:

        - ``session_description``: "not set"
        - ``session_start_time``: current UTC time

        ``identifier`` will be set in the following order of precedence:

        1. ``identifier`` in the inner dictionary
        2. ``nwbfile_kwargs["identifier"]`` (single-individual dataset only)
        3. individual name in the ``movement`` dataset

    processing_module_kwargs: dict[str, Any] or dict[str, dict[str, Any]], optional
        Keyword arguments for :class:`pynwb.base.ProcessingModule`.

        If ``processing_module_kwargs`` is a single dictionary, the same
        keyword arguments will be applied to all ProcessingModules.

        If ``processing_module_kwargs`` is a dictionary of dictionaries,
        the outer keys should correspond to individual names in the
        ``movement`` dataset, and the inner dictionaries will be passed as
        keyword arguments to the :class:`pynwb.file.ProcessingModule`
        constructor.

        The following arguments will have default values if not set:

        - ``name``: "behavior"
        - ``description``: "processed behavioral data"

    subject_kwargs : dict[str, Any] or dict[str, dict[str, Any]], optional
        Keyword arguments for :class:`pynwb.file.Subject`.

        If ``subject_kwargs`` is a single dictionary, the same keyword
        arguments will be applied to all Subjects except for ``subject_id``.

        If ``subject_kwargs`` is a dictionary of dictionaries, the outer keys
        should correspond to individual names in the ``movement`` dataset,
        and the inner dictionaries will be passed as keyword arguments to the
        :class:`pynwb.file.Subject` constructor.

        ``subject_id`` will be set in the following order of precedence:

        1. ``subject_id`` in the inner dictionary
        2. ``subject_kwargs["subject_id"]`` (single-individual dataset only)
        3. individual name in the ``movement`` dataset

    pose_estimation_series_kwargs : dict[str, Any] or dict[str, dict[str, Any]], optional
        Keyword arguments for ``ndx_pose.PoseEstimationSeries`` [1]_.

        If ``pose_estimation_series_kwargs`` is a single dictionary, the same
        keyword arguments will be applied to all PoseEstimationSeries objects.

        If ``pose_estimation_series_kwargs`` is a dictionary of dictionaries,
        the outer keys should correspond to keypoint names in the
        ``movement`` dataset, and the inner dictionaries will be passed as
        keyword arguments to the ``ndx_pose.PoseEstimationSeries`` constructor.

        The following arguments will be set based on the dataset and cannot
        be overwritten:

        - ``data``: position data for the keypoint
        - ``confidence``: confidence data for the keypoint
        - ``timestamps``: time data for the keypoint

        The following arguments will have default values if not set:

        - ``unit``: "pixels"
        - ``reference_frame``: "(0,0,0) corresponds to ..."

        ``name`` will be set in the following order of precedence:

        1. ``name`` in the inner dictionary
        2. ``pose_estimation_series_kwargs["name"]`` (single-keypoint
           dataset only)
        3. keypoint name in the ``movement`` dataset

    pose_estimation_kwargs : dict[str, Any] or dict[str, dict[str, Any]], optional
        Keyword arguments for ``ndx_pose.PoseEstimation`` [1]_.

        If ``pose_estimation_kwargs`` is a single dictionary, the same
        keyword arguments will be applied to all PoseEstimation objects.

        If ``pose_estimation_kwargs`` is a dictionary of dictionaries,
        the outer keys should correspond to individual names in the
        ``movement`` dataset, and the inner dictionaries will be passed as
        keyword arguments to the ``ndx_pose.PoseEstimation`` constructor.

        The following arguments cannot be overwritten:

        - ``pose_estimation_series``: list of PoseEstimationSeries objects
        - ``skeleton``: Skeleton object

        The following arguments will have default values if not set:

        - ``source_software``: ``source_software`` attribute from the
          ``movement`` dataset
        - ``description``: "Estimated positions of <keypoints> for
          <individual> using <source_software>."

        If specified, ``name`` will be set in the following
        order of precedence:

        1. ``name`` in the inner dictionary
        2. ``pose_estimation_kwargs["name"]`` (single-individual dataset only)
        3. individual name in the ``movement`` dataset

    skeleton_kwargs : dict[str, Any] or dict[str, dict[str, Any]], optional
        Keyword arguments for ``ndx_pose.Skeleton`` [1]_.

        If ``skeleton_kwargs`` is a single dictionary, the same
        keyword arguments will be applied to all Skeleton objects.

        If ``skeleton_kwargs`` is a dictionary of dictionaries,
        the outer keys should correspond to individual names in the
        ``movement`` dataset, and the inner dictionaries will be passed as
        keyword arguments to the ``ndx_pose.Skeleton`` constructor.

        The following arguments cannot be overwritten:

        - ``subject``: :class:`pynwb.file.Subject` created for the individual
          using ``subject_kwargs``

        The following arguments will have default values if not set:

        - ``name``: "<individual>_skeleton"
        - ``nodes``: list of keypoint names in the dataset

        ``name`` will be set in the following order of precedence:

        1. ``name`` in the inner dictionary
        2. ``skeleton_kwargs["name"]`` (single-individual dataset only)
        3. individual name in the ``movement`` dataset

    References
    ----------
    .. [1] https://github.com/rly/ndx-pose

    See Also
    --------
    movement.io.save_poses.to_nwb_file
        Example usage of this class to save a ``movement`` dataset
        to an NWB file.

    """  # noqa: E501

    nwbfile_kwargs: ConfigKwargsType = _safe_dict_field()
    processing_module_kwargs: ConfigKwargsType = _safe_dict_field()
    subject_kwargs: ConfigKwargsType = _safe_dict_field()
    pose_estimation_series_kwargs: ConfigKwargsType = _safe_dict_field()
    pose_estimation_kwargs: ConfigKwargsType = _safe_dict_field()
    skeleton_kwargs: ConfigKwargsType = _safe_dict_field()
    DEFAULT_NWBFILE_KWARGS = dict(session_description="not set")
    DEFAULT_PROCESSING_MODULE_KWARGS = dict(
        name="behavior", description="processed behavioral data"
    )
    DEFAULT_POSE_ESTIMATION_SERIES_KWARGS = dict(
        reference_frame="(0,0,0) corresponds to ...", unit="pixels"
    )

    def _resolve_kwargs(
        self,
        attr_name: str,
        entity: str | None,
        entity_type: str,
        id_key: str,
        prioritise_entity: bool = False,
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
        entity : str or None
            Individual or keypoint name.
        entity_type : str or None
            Type of entity (i.e. "individual", "keypoint"). Used in error
            or warning messages.
        id_key : str
            The key in ``cfg`` corresponding to the entity identifier/name
            (e.g. ``identifier``, ``subject_id``, ``name``) to be set in the
            returned dictionary.
        prioritise_entity : bool, optional
            Flag indicating whether ``entity`` should take precedence over
            the ``id_key`` when the attribute is a shared config
            (i.e. a single dictionary). Default is False.

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
                    f"setting '{entity}' as {id_key}."
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

    def _resolve_nwbfile_kwargs(
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
            Flag indicating whether ``individual`` should take precedence over
            the ``identifier`` in shared ``nwbfile_kwargs``. Default is True.

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
        if "session_start_time" not in kwargs:
            logger.warning(
                "No session_start_time provided in nwbfile_kwargs; "
                "using current UTC time as default."
            )
            kwargs["session_start_time"] = datetime.datetime.now(datetime.UTC)
        return kwargs

    def _resolve_processing_module_kwargs(
        self, individual: str | None = None
    ) -> dict[str, Any]:
        """Resolve the keyword arguments for :class:`pynwb.base.ProcessingModule`.

        Parameters
        ----------
        individual : str, optional
            Individual name. If provided, the method will attempt to retrieve
            individual-specific settings or fall back to shared or default
            settings.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to be passed to :class:`pynwb.base.ProcessingModule`.

        """  # noqa: E501
        kwargs = self._resolve_kwargs(
            attr_name="processing_module_kwargs",
            entity=individual,
            entity_type="individual",
            id_key="name",
        )
        if kwargs.get("name") in (individual, None):
            kwargs["name"] = self.DEFAULT_PROCESSING_MODULE_KWARGS["name"]
        return kwargs

    def _resolve_subject_kwargs(
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
            Flag indicating whether ``individual`` should take precedence over
            the ``subject_id`` in shared ``subject_kwargs``. Default is True.

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

    def _resolve_pose_estimation_series_kwargs(
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
            Flag indicating whether ``keypoint`` should take precedence over
            the ``name`` in shared ``pose_estimation_series_kwargs``.
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

    def _resolve_pose_estimation_kwargs(
        self,
        individual: str | None = None,
        prioritise_individual: bool = True,
        defaults: dict | None = None,
    ) -> dict[str, Any]:
        """Resolve the keyword arguments for ``ndx_pose.PoseEstimation``.

        Parameters
        ----------
        individual : str, optional
            Individual name. If provided, the method will attempt to retrieve
            individual-specific settings or fall back to shared or default
            settings.
        prioritise_individual: bool, optional
            Flag indicating whether ``individual`` should take precedence over
            the ``name`` in shared ``pose_estimation_kwargs``. Default is True.
        defaults : dict, optional
            Dataset-specific default values to be used.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to be passed to ``ndx_pose.PoseEstimation``.

        """
        kwargs = self._resolve_kwargs(
            attr_name="pose_estimation_kwargs",
            entity=individual,
            entity_type="individual",
            id_key="name",
            prioritise_entity=prioritise_individual,
        )
        if defaults is not None:
            for key, value in defaults.items():
                kwargs.setdefault(key, value)
        if not self._has_key(self.pose_estimation_kwargs, "name"):
            kwargs.pop("name", None)  # Use ndx_pose default
        return kwargs

    def _resolve_skeleton_kwargs(
        self,
        individual: str | None = None,
        prioritise_individual: bool = True,
        defaults: dict | None = None,
    ) -> dict[str, Any]:
        """Resolve the keyword arguments for ``ndx_pose.Skeleton``.

        Parameters
        ----------
        individual : str, optional
            Individual name. If provided, the method will attempt to retrieve
            individual-specific settings or fall back to shared or default
            settings.
        prioritise_individual: bool, optional
            Flag indicating whether ``individual`` should take precedence over
            the ``name`` in shared ``skeleton_kwargs``. Default is True.
        defaults : dict, optional
            Dataset-specific default values to be used.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to be passed to ``ndx_pose.Skeleton``.

        """
        kwargs = self._resolve_kwargs(
            attr_name="skeleton_kwargs",
            entity=individual,
            entity_type="individual",
            id_key="name",
            prioritise_entity=prioritise_individual,
        )
        if defaults is not None:
            for key, value in defaults.items():
                kwargs.setdefault(key, value)
        if not self._has_key(self.skeleton_kwargs, "name"):
            kwargs["name"] = (
                f"skeleton{f'_{individual}' if individual else ''}"
            )
        return kwargs

    @staticmethod
    def _is_per_entity_config(cfg: ConfigKwargsType) -> bool:
        return bool(cfg) and all(isinstance(v, dict) for v in cfg.values())

    @staticmethod
    def _has_key(cfg: ConfigKwargsType, key: str) -> bool:
        if isinstance(cfg, dict):
            if key in cfg:
                return True
            for sub_dict in cfg.values():
                if isinstance(sub_dict, dict) and key in sub_dict:
                    return True
        return False


def _ds_to_pose_and_skeletons(
    ds: xr.Dataset,
    config: NWBFileSaveConfig | None = None,
    subject: pynwb.file.Subject | None = None,
    from_multi_individual: bool = False,
) -> tuple[ndx_pose.PoseEstimation, ndx_pose.Skeletons]:
    """Create PoseEstimation and Skeletons objects from a ``movement`` dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        A single-individual ``movement`` poses dataset.
    config : movement.io.nwb.NWBFileSaveConfig
        Configuration object containing keyword arguments to customise
        the PoseEstimation and Skeletons objects created from the dataset.
        If None (default), default values will be used.
        See :class:`movement.io.nwb.NWBFileSaveConfig` for more details.
    subject : pynwb.file.Subject, optional
        Subject object to be linked in the Skeleton object.
    from_multi_individual : bool, optional
        Flag indicating whether ``ds`` originates from a multi-individual
        dataset. Passed to the ``NWBFileSaveConfig`` methods to determine
        whether to prioritise individual names in the dataset over ``name``
        in shared ``pose_estimation_kwargs`` and ``skeleton_kwargs``.
        Default is False.

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
    individual = ds.individuals.values.item()
    keypoints = ds.keypoints.values.tolist()
    # Convert timestamps to seconds if necessary
    timestamps = (
        ds.time.values
        if ds.time_unit == "seconds"
        else ds.time.values / getattr(ds, "fps", 1.0)
    )
    pose_estimation_series = [
        ndx_pose.PoseEstimationSeries(
            data=ds.sel(keypoints=keypoint).position.values,
            confidence=ds.sel(keypoints=keypoint).confidence.values,
            timestamps=timestamps,
            **(
                config._resolve_pose_estimation_series_kwargs(
                    keypoint, len(keypoints) > 1
                )
            ),
        )
        for keypoint in keypoints
    ]
    skeleton_list = [
        ndx_pose.Skeleton(
            subject=subject,
            **config._resolve_skeleton_kwargs(
                individual, from_multi_individual, {"nodes": keypoints}
            ),
        )
    ]
    skeletons = ndx_pose.Skeletons(skeletons=skeleton_list)
    # Group all PoseEstimationSeries into a PoseEstimation object
    description = (
        f"Estimated positions of {', '.join(keypoints)} for "
        f"{individual} using {ds.source_software}."
    )
    pose_estimation = ndx_pose.PoseEstimation(
        pose_estimation_series=pose_estimation_series,
        skeleton=skeleton_list[-1],
        **config._resolve_pose_estimation_kwargs(
            individual,
            from_multi_individual,
            {
                "description": description,
                "source_software": ds.source_software,
            },
        ),
    )
    return pose_estimation, skeletons


def _write_processing_module(
    nwb_file: pynwb.file.NWBFile,
    processing_module_kwargs: dict[str, Any],
    pose_estimation: ndx_pose.PoseEstimation,
    skeletons: ndx_pose.Skeletons,
) -> None:
    """Write behaviour processing data to an NWB file.

    PoseEstimation or Skeletons objects will be written to the specified
    ProcessingModule in the NWB file, formatted according to the
    ``ndx-pose`` NWB extension. If the module does not exist, it will be
    created. Existing objects in the NWB file will not be overwritten.

    Parameters
    ----------
    nwb_file : pynwb.file.NWBFile
        The NWBFile object to which the data will be added.
    processing_module_kwargs : dict[str, Any]
        Keyword arguments for the :class:`pynwb.base.ProcessingModule` in the
        NWB file. The ``name`` key will be used to determine the
        ProcessingModule to which the data will be added.
        If the ProcessingModule does not exist, it will be created with these
        keyword arguments.
    pose_estimation : ndx_pose.PoseEstimation
        PoseEstimation object containing the pose data for an individual.
    skeletons : ndx_pose.Skeletons
        Skeletons object containing the skeleton data for an individual.

    """

    def add_to_processing_module(obj, obj_name: str):
        try:
            processing_module.add(obj)
            logger.debug(f"Added {obj_name} object to NWB file.")
        except ValueError:
            logger.warning(f"{obj_name} object already exists. Skipping...")

    processing_module_name = processing_module_kwargs.get("name")
    processing_module = nwb_file.processing.get(processing_module_name)
    if processing_module is None:
        processing_module = nwb_file.create_processing_module(
            **processing_module_kwargs
        )
        logger.debug(
            f"Created {processing_module_name} processing module in NWB file."
        )
    else:
        logger.debug(
            f"Using existing {processing_module_name} processing module."
        )
    add_to_processing_module(skeletons, "Skeletons")
    add_to_processing_module(pose_estimation, "PoseEstimation")
