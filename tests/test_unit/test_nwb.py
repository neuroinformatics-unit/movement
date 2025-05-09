import datetime
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import ndx_pose
import pytest

from movement.io.nwb import NWBFileSaveConfig, _ds_to_pose_and_skeleton_objects


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


class TestNWBFileSaveConfig:
    """Test the NWBFileSaveConfig class."""

    SESSION_START_TIME = datetime.datetime.now(datetime.UTC)

    @pytest.mark.parametrize(
        "nwbfile_kwargs, individual, expected",
        [
            (
                None,
                None,
                does_not_raise(
                    {
                        "session_description": "not set",
                        "identifier": "not set",
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
                pytest.raises(
                    ValueError, match=".*no individual was provided."
                ),
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
        ],
        ids=[
            "no args: default identifier",
            "ind: ind as identifier",
            "shared kwargs: kwarg as identifier",
            "shared kwargs + ind: ind as identifier",
            "kwargs per ind: error (multiple keys but no ind provided)",
            "kwargs per ind (assume key is ind): warn; kwarg as identifier",
            "kwargs per ind + ind: kwarg as identifier",
            "kwargs per ind (w/o identifier) + ind: ind as identifier",
            "kwargs per ind + ind (not in kwargs): warn; ind as identifier",
        ],
    )
    def test_resolve_nwbfile_kwargs(
        self, nwbfile_kwargs, individual, expected, request, caplog
    ):
        """Test NWBFileSaveConfig create_nwbfile with nwbfile_kwargs."""
        with patch("datetime.datetime") as mock_datetime, expected as context:
            mock_datetime.now.return_value = self.SESSION_START_TIME
            config = NWBFileSaveConfig(nwbfile_kwargs=nwbfile_kwargs)
            actual_kwargs = config.resolve_nwbfile_kwargs(individual)
            expected_kwargs = context or nwbfile_kwargs
            assert actual_kwargs == expected_kwargs
            if "warn; kwarg as identifier" in request.node.callspec.id:
                assert "Assuming 'id_0'" in caplog.messages[0]
            if "warn; ind as identifier" in request.node.callspec.id:
                assert "'id_not_in_kwargs' not found" in caplog.messages[0]

    @pytest.mark.parametrize(
        "subject_kwargs, individual, expected",
        [
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
                pytest.raises(
                    ValueError, match=".*no individual was provided."
                ),
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
        ],
        ids=[
            "no args: defaults",
            "ind: ind as id",
            "shared kwargs: kwarg as id",
            "shared kwargs + ind: ind as id",
            "kwargs per ind: error (multiple keys but no ind provided)",
            "kwargs per ind (assume key is ind): warn; kwarg as id",
            "kwargs per ind + ind: kwarg as id",
            "kwargs per ind (w/o id) + ind: ind as id",
            "kwargs per ind + ind (not in kwargs): warn; ind as id",
        ],
    )
    def test_resolve_subject_kwargs(
        self, subject_kwargs, individual, expected, request, caplog
    ):
        """Test NWBFileSaveConfig create_nwbfile with nwbfile_kwargs."""
        with expected as expected_kwargs:
            config = NWBFileSaveConfig(subject_kwargs=subject_kwargs)
            actual_kwargs = config.resolve_subject_kwargs(individual)
            assert actual_kwargs == expected_kwargs
            if "warn; kwarg as id" in request.node.callspec.id:
                assert "Assuming 'id_0'" in caplog.messages[0]
            if "warn; ind as id" in request.node.callspec.id:
                assert "'id_not_in_kwargs' not found" in caplog.messages[0]
