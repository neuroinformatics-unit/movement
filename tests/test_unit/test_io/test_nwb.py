import datetime
from contextlib import AbstractContextManager as ContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import patch

import ndx_pose
import pytest
from pynwb.file import Subject

from movement.io.nwb import (
    NWBFileSaveConfig,
    _ds_to_pose_and_skeletons,
    _write_processing_module,
)


@pytest.mark.parametrize(
    "selection_fn, expected_subject_id",
    [
        (lambda ds: ds.sel(individual="id_0"), "id_0"),
        (
            lambda ds: ds.sel(individual=ds.individual[0]),
            "id_0",
        ),  # Should still work if only one individual
    ],
    ids=["single_individual_selected", "single_individual_implicit"],
)
def test_ds_to_pose_and_skeletons(
    valid_poses_dataset, selection_fn, expected_subject_id
):
    """Test the conversion of a valid poses dataset to
    ``ndx_pose`` PoseEstimation and Skeletons.
    """
    # Use single-individual dataset for simplicity
    ds = selection_fn(valid_poses_dataset)
    pose_estimation, skeletons = _ds_to_pose_and_skeletons(
        ds,
        subject=Subject(subject_id=expected_subject_id),
    )
    assert isinstance(pose_estimation, ndx_pose.PoseEstimation)
    assert isinstance(skeletons, ndx_pose.Skeletons)
    assert (
        set(valid_poses_dataset.keypoint.values)
        == pose_estimation.pose_estimation_series.keys()
    )
    assert {f"skeleton_{expected_subject_id}"} == skeletons.skeletons.keys()
    assert (
        skeletons.skeletons[
            f"skeleton_{expected_subject_id}"
        ].subject.subject_id
        == expected_subject_id
    )


def test_ds_to_pose_and_skeletons_invalid(valid_poses_dataset):
    """Test the conversion of a poses dataset with more than one
    individual to ``ndx_pose`` PoseEstimation and Skeletons raises
    an error.
    """
    with pytest.raises(
        ValueError,
        match=".*must contain only one individual.*",
    ):
        _ds_to_pose_and_skeletons(valid_poses_dataset)


def test_write_processing_module(nwbfile_object, caplog):
    """Test that writing to an NWBFile with existing "behavior"
    processing module, Skeletons, and PoseEstimation will
    not overwrite them.
    """
    _write_processing_module(
        nwbfile_object(),
        {"name": "behavior"},
        ndx_pose.PoseEstimation(),
        ndx_pose.Skeletons(),
    )
    assert {
        "Using existing behavior processing module.",
        "Skeletons object already exists. Skipping...",
        "PoseEstimation object already exists. Skipping...",
    } == set(caplog.messages)


NWBFileSaveConfigTestCase = tuple[
    dict[str, Any] | None, str | None, ContextManager[Any]
]


class TestNWBFileSaveConfig:
    """Test the NWBFileSaveConfig class."""

    SESSION_START_TIME = datetime.datetime.now(datetime.UTC)

    # --- Parameter sets ---
    # Generic case IDs for nwbfile, subject, pose_estimation,
    # pose_estimation_series, and skeletons kwargs, where the
    # entity (individual/keypoint) name can either be extracted
    # from the config object, the specified entity arg, or defaults.
    CASE_IDS_GENERIC = [
        "no args: default id",
        "entity: entity as id",
        "shared kwargs: kwarg as id",
        "shared kwargs + entity: entity as id",
        "kwargs per entity: error (multiple keys but no entity provided)",
        "kwargs per entity (assume key is entity): warn; kwarg as id",
        "kwargs per entity + entity: kwarg as id",
        "kwargs per entity (w/o id) + entity: entity as id",
        "kwargs per entity + entity (not in kwargs): warn; entity as id",
    ]
    # Special case IDs for processing module kwargs as the module name
    # is expected to either be user-specified via the config or the
    # default "behavior" (i.e. never the individual name).
    CASE_IDS_PROCESSING_MODULE = [
        "no args: default id",
        "entity: default as id",
        "shared kwargs: kwarg as id",
        "shared kwargs + entity: kwargs as id",
        "kwargs per entity: error (multiple keys but no entity provided)",
        "kwargs per entity (assume key is entity): warn; kwarg as id",
        "kwargs per entity + entity: kwarg as id",
        "kwargs per entity (w/o id) + entity: default as id",
        "kwargs per entity + entity (not in kwargs): warn; default as id",
    ]
    nwbfile_kwargs_params: list[NWBFileSaveConfigTestCase] = [
        (
            None,
            None,
            does_not_raise(
                {
                    "session_description": "not set",
                    "session_start_time": SESSION_START_TIME,
                }
            ),
        ),
        (
            None,
            "id_0",
            does_not_raise(
                {
                    "session_description": "not set",
                    "identifier": "id_0",
                    "session_start_time": SESSION_START_TIME,
                }
            ),
        ),
        (
            {
                "session_description": "subj0 session",
                "identifier": "subj0",
                "session_start_time": SESSION_START_TIME,
            },
            None,
            does_not_raise(None),  # Same as input kwargs
        ),
        (
            {
                "session_description": "subj0 session",
                "identifier": "subj0",
                "session_start_time": SESSION_START_TIME,
            },
            "id_0",
            does_not_raise(
                {
                    "session_description": "subj0 session",
                    "identifier": "id_0",
                    "session_start_time": SESSION_START_TIME,
                }
            ),
        ),
        (
            {
                "id_0": {
                    "session_description": "subj0 session",
                    "identifier": "subj0",
                },
                "id_1": {
                    "session_description": "subj1 session",
                    "identifier": "subj1",
                },
            },
            None,
            pytest.raises(ValueError, match=".*no individual was provided."),
        ),
        (
            {
                "id_0": {
                    "session_description": "subj0 session",
                    "identifier": "subj0",
                },
            },
            None,
            does_not_raise(
                {
                    "session_description": "subj0 session",
                    "identifier": "subj0",
                    "session_start_time": SESSION_START_TIME,
                }
            ),
        ),
        (
            {
                "id_0": {
                    "session_description": "subj0 session",
                    "identifier": "subj0",
                },
            },
            "id_0",
            does_not_raise(
                {
                    "session_description": "subj0 session",
                    "identifier": "subj0",
                    "session_start_time": SESSION_START_TIME,
                }
            ),
        ),
        (
            {
                "id_0": {
                    "session_description": "subj0 session",
                },
            },
            "id_0",
            does_not_raise(
                {
                    "session_description": "subj0 session",
                    "identifier": "id_0",
                    "session_start_time": SESSION_START_TIME,
                }
            ),
        ),
        (
            {
                "id_0": {
                    "session_description": "subj0 session",
                },
            },
            "id_not_in_kwargs",
            does_not_raise(
                {
                    "session_description": "not set",
                    "identifier": "id_not_in_kwargs",
                    "session_start_time": SESSION_START_TIME,
                }
            ),
        ),
    ]
    processing_module_kwargs_params: list[NWBFileSaveConfigTestCase] = [
        (
            None,
            None,
            does_not_raise(
                {
                    "name": "behavior",
                    "description": "processed behavioral data",
                }
            ),
        ),
        (
            None,
            "id_0",
            does_not_raise(
                {
                    "name": "behavior",
                    "description": "processed behavioral data",
                }
            ),
        ),
        (
            {"name": "behaviour"},
            None,
            does_not_raise(
                {
                    "name": "behaviour",
                    "description": "processed behavioral data",
                }
            ),
        ),
        (
            {"name": "behaviour"},
            "id_0",
            does_not_raise(
                {
                    "name": "behaviour",
                    "description": "processed behavioral data",
                }
            ),
        ),
        (
            {"id_0": {"name": "behaviour0"}, "id_1": {"name": "behaviour1"}},
            None,
            pytest.raises(ValueError, match=".*no individual was provided."),
        ),
        (
            {"id_0": {"name": "behaviour0"}},
            None,
            does_not_raise(
                {
                    "name": "behaviour0",
                    "description": "processed behavioral data",
                }
            ),
        ),
        (
            {"id_0": {"name": "behaviour0"}},
            "id_0",
            does_not_raise(
                {
                    "name": "behaviour0",
                    "description": "processed behavioral data",
                }
            ),
        ),
        (
            {"id_0": {"description": "processed behav data"}},
            "id_0",
            does_not_raise(
                {"name": "behavior", "description": "processed behav data"}
            ),
        ),
        (
            {"id_0": {"name": "behaviour0"}},
            "id_not_in_kwargs",
            does_not_raise(
                {
                    "name": "behavior",
                    "description": "processed behavioral data",
                }
            ),
        ),
    ]
    subject_kwargs_params: list[NWBFileSaveConfigTestCase] = [
        (None, None, does_not_raise({})),
        (None, "id_0", does_not_raise({"subject_id": "id_0"})),
        (
            {"age": "P90D", "subject_id": "subj0"},
            None,
            does_not_raise({"age": "P90D", "subject_id": "subj0"}),
        ),
        (
            {"age": "P90D", "subject_id": "subj0"},
            "id_0",
            does_not_raise({"age": "P90D", "subject_id": "id_0"}),
        ),
        (
            {
                "id_0": {"age": "P90D", "subject_id": "subj0"},
                "id_1": {"age": "P91D", "subject_id": "subj1"},
            },
            None,
            pytest.raises(ValueError, match=".*no individual was provided."),
        ),
        (
            {"id_0": {"age": "P90D", "subject_id": "subj0"}},
            None,
            does_not_raise({"age": "P90D", "subject_id": "subj0"}),
        ),
        (
            {"id_0": {"age": "P90D", "subject_id": "subj0"}},
            "id_0",
            does_not_raise({"age": "P90D", "subject_id": "subj0"}),
        ),
        (
            {"id_0": {"age": "P90D"}},
            "id_0",
            does_not_raise({"age": "P90D", "subject_id": "id_0"}),
        ),
        (
            {"id_0": {"age": "P90D"}},
            "id_not_in_kwargs",
            does_not_raise({"subject_id": "id_not_in_kwargs"}),
        ),
    ]
    pose_estimation_series_kwargs_params: list[NWBFileSaveConfigTestCase] = [
        (
            None,
            None,
            does_not_raise(
                {
                    "reference_frame": "(0,0,0) corresponds to ...",
                    "unit": "pixels",
                }
            ),
        ),
        (
            None,
            "centroid",
            does_not_raise(
                {
                    "reference_frame": "(0,0,0) corresponds to ...",
                    "name": "centroid",
                    "unit": "pixels",
                }
            ),
        ),
        (
            {"name": "anchor"},
            None,
            does_not_raise(
                {
                    "reference_frame": "(0,0,0) corresponds to ...",
                    "name": "anchor",
                    "unit": "pixels",
                }
            ),
        ),
        (
            {"name": "anchor"},
            "centroid",
            does_not_raise(
                {
                    "reference_frame": "(0,0,0) corresponds to ...",
                    "name": "centroid",
                    "unit": "pixels",
                }
            ),
        ),
        (
            {"centroid": {"name": "anchor"}, "left": {"name": "left_ear"}},
            None,
            pytest.raises(
                ValueError,
                match=".*no keypoint was provided.",
            ),
        ),
        (
            {"centroid": {"name": "anchor"}},
            None,
            does_not_raise(
                {
                    "reference_frame": "(0,0,0) corresponds to ...",
                    "name": "anchor",
                    "unit": "pixels",
                }
            ),
        ),
        (
            {"centroid": {"name": "anchor"}},
            "centroid",
            does_not_raise(
                {
                    "reference_frame": "(0,0,0) corresponds to ...",
                    "name": "anchor",
                    "unit": "pixels",
                }
            ),
        ),
        (
            {"centroid": {"description": "anchor part"}},
            "centroid",
            does_not_raise(
                {
                    "reference_frame": "(0,0,0) corresponds to ...",
                    "description": "anchor part",
                    "name": "centroid",
                    "unit": "pixels",
                }
            ),
        ),
        (
            {"centroid": {"name": "anchor"}},
            "name_not_in_kwargs",
            does_not_raise(
                {
                    "reference_frame": "(0,0,0) corresponds to ...",
                    "name": "name_not_in_kwargs",
                    "unit": "pixels",
                }
            ),
        ),
    ]
    pose_estimation_kwargs_params: list[NWBFileSaveConfigTestCase] = [
        (
            None,
            None,
            does_not_raise({}),
        ),
        (
            None,
            "subj0",
            does_not_raise({}),
        ),
        ({"name": "subj0"}, None, does_not_raise({"name": "subj0"})),
        (
            {"name": "subj0"},
            "id_0",
            does_not_raise({"name": "id_0"}),
        ),
        (
            {"id_0": {"name": "subj0"}, "id_1": {"name": "subj1"}},
            None,
            pytest.raises(ValueError, match=".*no individual was provided."),
        ),
        (
            {"id_0": {"name": "subj0"}},
            None,
            does_not_raise({"name": "subj0"}),
        ),
        (
            {"id_0": {"name": "subj0"}},
            "id_0",
            does_not_raise({"name": "subj0"}),
        ),
        (
            {"id_0": {"name": "subj0"}},
            "id_not_in_kwargs",
            does_not_raise({"name": "id_not_in_kwargs"}),
        ),
    ]
    skeleton_kwargs_params: list[NWBFileSaveConfigTestCase] = [
        (
            None,
            None,
            does_not_raise({"name": "skeleton"}),
        ),
        (
            None,
            "id_0",
            does_not_raise({"name": "skeleton_id_0"}),
        ),
        ({"name": "id_0"}, None, does_not_raise({"name": "id_0"})),
        (
            {"name": "id_0"},
            "id_1",
            does_not_raise({"name": "id_1"}),
        ),
        (
            {"id_0": {"name": "id_0"}, "id_1": {"name": "id_1"}},
            None,
            pytest.raises(ValueError, match=".*no individual was provided."),
        ),
        (
            {"id_0": {"name": "id_0"}},
            None,
            does_not_raise({"name": "id_0"}),
        ),
        (
            {"id_0": {"name": "id_0_skeleton"}},
            "id_0",
            does_not_raise({"name": "id_0_skeleton"}),
        ),
        (
            {"id_0": {"name": "id_0"}},
            "id_not_in_kwargs",
            does_not_raise({"name": "id_not_in_kwargs"}),
        ),
    ]
    ATTR_PARAMS = {
        "nwbfile_kwargs": nwbfile_kwargs_params,
        "processing_module_kwargs": processing_module_kwargs_params,
        "subject_kwargs": subject_kwargs_params,
        "pose_estimation_series_kwargs": pose_estimation_series_kwargs_params,
        "pose_estimation_kwargs": pose_estimation_kwargs_params,
        "skeleton_kwargs": skeleton_kwargs_params,
    }

    combined_params = []
    combined_ids = []

    for attr, param_list in ATTR_PARAMS.items():
        case_ids = (
            CASE_IDS_PROCESSING_MODULE
            if attr == "processing_module_kwargs"
            else CASE_IDS_GENERIC
        )
        for i, (kwargs, entity, expected) in enumerate(param_list):
            combined_params.append((attr, kwargs, entity, expected))
            combined_ids.append(f"{attr}-{case_ids[i]}")

    @pytest.mark.parametrize(
        "attr, input_kwargs, entity, expected_context",
        combined_params,
        ids=combined_ids,
    )
    def test_resolve_kwargs(
        self, attr, input_kwargs, entity, expected_context, caplog, request
    ):
        """Test resolving a specific attribute of NWBFileSaveConfig."""
        with (
            patch("datetime.datetime") as mock_datetime,
            expected_context as expected,
        ):
            mock_datetime.now.return_value = self.SESSION_START_TIME
            config = NWBFileSaveConfig(**{attr: input_kwargs})
            resolver = getattr(config, f"_resolve_{attr}")
            actual = resolver(entity)
            assert actual == expected or input_kwargs
        case_id = request.node.callspec.id
        if "warn; kwarg as" in case_id:
            assert (
                f"Assuming '{entity or next(iter(input_kwargs))}'"
                in caplog.messages[0]
            )
        elif "warn; ind as" in case_id or "warn; kp as" in case_id:
            assert f"'{entity}' not found" in caplog.messages[0]

    def test_warning_if_session_start_time_not_provided(self, caplog):
        """Test that not setting ``session_start_time`` in ``nwb_file_kwargs``
        results in a warning message about using the current UTC time
        as default.
        """
        NWBFileSaveConfig()._resolve_nwbfile_kwargs()
        assert "using current UTC time as default" in caplog.messages[0]
