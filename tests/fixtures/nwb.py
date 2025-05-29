"""NWB file and NWBFileSaveConfig fixtures."""

import datetime

import numpy as np
import pytest
from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

from movement.io.nwb import NWBFileSaveConfig


# ---------------- NWB file fixtures ----------------------------
@pytest.fixture
def nwb_file(nwbfile_object, tmp_path):
    """Return the file path for a valid NWB poses file."""

    def _nwb_file(**kwargs):
        """Create a valid NWB file with poses.
        ``kwargs`` are passed to ``create_pose_estimation_series``.
        """
        file_path = tmp_path / "test_pose.nwb"
        with NWBHDF5IO(file_path, mode="w") as io:
            io.write(nwbfile_object(**kwargs))
        return file_path

    return _nwb_file


@pytest.fixture
def nwbfile_object(rng):
    """Return an NWBFile object containing poses for
    a single individual with three keypoints and the associated
    skeleton object, as well as a camera device.
    """

    def _nwbfile_object(**kwargs):
        """Create an NWBFile object with poses.
        ``kwargs`` are passed to ``create_pose_estimation_series``.
        """
        identifier = "subj1"
        nwb_file_obj = NWBFile(
            session_description="session_description",
            identifier=identifier,
            session_start_time=datetime.datetime.now(datetime.UTC),
        )
        subject = Subject(subject_id=identifier, species="Mus musculus")
        nwb_file_obj.subject = subject
        keypoints = ["front_left_paw", "body", "front_right_paw"]
        skeleton = Skeleton(
            name="subj1_skeleton",
            nodes=keypoints,
            edges=np.array([[0, 1], [1, 2]], dtype="uint8"),
            subject=subject,
        )
        skeletons = Skeletons(skeletons=[skeleton])
        camera1 = nwb_file_obj.create_device(
            name="camera1",
            description="camera for recording behavior",
            manufacturer="my manufacturer",
        )
        pose_estimation_series = []
        for keypoint in keypoints:
            pose_estimation_series.append(
                create_pose_estimation_series(rng, keypoint, **kwargs)
            )
        pose_estimation = PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=pose_estimation_series,
            description=(
                "Estimated positions of front paws of subj1 using DeepLabCut."
            ),
            original_videos=["path/to/camera1.mp4"],
            labeled_videos=["path/to/camera1_labeled.mp4"],
            dimensions=np.array(
                [[640, 480]], dtype="uint16"
            ),  # pixel dimensions of the video
            devices=[camera1],
            scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
            source_software="DeepLabCut",
            source_software_version="2.3.8",
            skeleton=skeleton,  # link to the skeleton object
        )
        behavior_pm = nwb_file_obj.create_processing_module(
            name="behavior",
            description="processed behavioral data",
        )
        behavior_pm.add(skeletons)
        behavior_pm.add(pose_estimation)
        return nwb_file_obj

    return _nwbfile_object


def create_pose_estimation_series(
    rng, keypoint, starting_time=None, rate=None, timestamps=None
):
    """Create a PoseEstimationSeries object for a keypoint,
    providing either ``rate`` and ``starting_time`` or just ``timestamps``.
    If none of these are provided, default ``timestamps`` are generated.
    """
    n_frames = 100
    n_dims = 2  # 2D (can also be 3D)
    if timestamps is not None:
        rate = None
    if timestamps is None and rate is None:
        timestamps = np.arange(n_frames) / 10.0  # assuming fps=10.0
    return PoseEstimationSeries(
        name=keypoint,
        description="Marker placed around fingers of front left paw.",
        data=rng.random((n_frames, n_dims)),
        unit="pixels",
        reference_frame="(0,0,0) corresponds to ...",
        timestamps=timestamps,
        starting_time=starting_time,
        rate=rate,
        confidence=np.ones((n_frames,)),  # confidence in each frame
        confidence_definition="Softmax output of the DNN.",
    )


# ---------------- NWBFileSaveConfig fixtures ----------------------------
@pytest.fixture
def shared_nwb_config():
    """Fixture to provide a shared NWBFileSaveConfig."""
    return NWBFileSaveConfig(
        nwbfile_kwargs={
            "session_description": "test session",
            "identifier": "subj0",
        },
        processing_module_kwargs={
            "description": "processed behav for test session",
        },
        subject_kwargs={"age": "P90D", "subject_id": "subj0"},
        pose_estimation_series_kwargs={
            "reference_frame": "(0,0) is ...",
            "name": "anchor",
        },
        pose_estimation_kwargs={"name": "subj0", "source_software": "other"},
        skeleton_kwargs={
            "name": "skeleton0",
            "nodes": ["anchor", "left_ear", "right_ear"],
        },
    )


@pytest.fixture
def per_entity_nwb_config():
    """Fixture to provide a per-entity (individual and keypoint)
    NWBFileSaveConfig.
    """
    return NWBFileSaveConfig(
        nwbfile_kwargs={
            "id_0": {
                "session_description": "session subj0",
                "identifier": "subj0",
            },
            "id_1": {
                "session_description": "session subj1",
                "identifier": "subj1",
            },
        },
        processing_module_kwargs={
            "id_0": {
                "description": "processed behav for subj0",
            },
            "id_1": {
                "description": "processed behav for subj1",
            },
        },
        subject_kwargs={
            "id_0": {"age": "P90D", "subject_id": "subj0"},
            "id_1": {"age": "P91D", "subject_id": "subj1"},
        },
        pose_estimation_series_kwargs={
            "centroid": {"name": "anchor"},
            "left": {"name": "left_ear"},
        },
        pose_estimation_kwargs={
            "id_0": {"name": "subj0", "source_software": "other0"},
            "id_1": {"name": "subj1", "source_software": "other1"},
        },
        skeleton_kwargs={
            "id_0": {"nodes": ["node1", "node2", "node3"]},
            "id_1": {"nodes": ["node4", "node5", "node6"]},
        },
    )
