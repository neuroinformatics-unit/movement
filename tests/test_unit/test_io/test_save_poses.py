from contextlib import nullcontext as does_not_raise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import DATA_PATHS

from movement.io import load_poses, save_poses

output_files = [
    {
        "file_fixture": "fake_h5_file",
        "to_dlc_file_expected_exception": pytest.raises(FileExistsError),
        "to_sleap_file_expected_exception": pytest.raises(FileExistsError),
        "to_lp_file_expected_exception": pytest.raises(FileExistsError),
        # invalid file path
    },
    {
        "file_fixture": "directory",
        "to_dlc_file_expected_exception": pytest.raises(IsADirectoryError),
        "to_sleap_file_expected_exception": pytest.raises(IsADirectoryError),
        "to_lp_file_expected_exception": pytest.raises(IsADirectoryError),
        # invalid file path
    },
    {
        "file_fixture": "wrong_extension_new_file",
        "to_dlc_file_expected_exception": pytest.raises(ValueError),
        "to_sleap_file_expected_exception": pytest.raises(ValueError),
        "to_lp_file_expected_exception": pytest.raises(ValueError),
        # invalid file path
    },
    {
        "file_fixture": "new_csv_file",
        "to_dlc_file_expected_exception": does_not_raise(),
        "to_sleap_file_expected_exception": pytest.raises(ValueError),
        "to_lp_file_expected_exception": does_not_raise(),
        # valid file path for dlc and lp, invalid for sleap
    },
    {
        "file_fixture": "new_h5_file",
        "to_dlc_file_expected_exception": does_not_raise(),
        "to_sleap_file_expected_exception": does_not_raise(),
        "to_lp_file_expected_exception": pytest.raises(ValueError),
        # valid file path for dlc and sleap, invalid for lp
    },
]

invalid_poses_datasets_and_exceptions = [
    ("not_a_dataset", TypeError),
    ("empty_dataset", ValueError),
    ("missing_var_poses_dataset", ValueError),
    ("missing_dim_poses_dataset", ValueError),
]


@pytest.fixture(params=output_files)
def output_file_params(request):
    """Return a dictionary containing parameters for testing saving
    valid pose datasets to DeepLabCut- or SLEAP-style files.
    """
    return request.param


@pytest.mark.parametrize(
    "ds, expected_exception",
    [
        (np.array([1, 2, 3]), pytest.raises(TypeError)),  # incorrect type
        (
            load_poses.from_dlc_file(
                DATA_PATHS.get("DLC_single-wasp.predictions.h5")
            ),
            does_not_raise(),
        ),  # valid dataset
        (
            load_poses.from_dlc_file(
                DATA_PATHS.get("DLC_two-mice.predictions.csv")
            ),
            does_not_raise(),
        ),  # valid dataset
        (
            load_poses.from_sleap_file(
                DATA_PATHS.get("SLEAP_single-mouse_EPM.analysis.h5")
            ),
            does_not_raise(),
        ),  # valid dataset
        (
            load_poses.from_sleap_file(
                DATA_PATHS.get(
                    "SLEAP_three-mice_Aeon_proofread.predictions.slp"
                )
            ),
            does_not_raise(),
        ),  # valid dataset
        (
            load_poses.from_lp_file(
                DATA_PATHS.get("LP_mouse-face_AIND.predictions.csv")
            ),
            does_not_raise(),
        ),  # valid dataset
    ],
)
def test_to_dlc_style_df(ds, expected_exception):
    """Test that converting a valid/invalid xarray dataset to
    a DeepLabCut-style pandas DataFrame returns the expected result.
    """
    with expected_exception as e:
        df = save_poses.to_dlc_style_df(ds, split_individuals=False)
        if e is None:  # valid input
            assert isinstance(df, pd.DataFrame)
            assert isinstance(df.columns, pd.MultiIndex)
            assert df.columns.names == [
                "scorer",
                "individuals",
                "bodyparts",
                "coords",
            ]


def test_to_dlc_file_valid_dataset(
    output_file_params, valid_poses_dataset, request
):
    """Test that saving a valid pose dataset to a valid/invalid
    DeepLabCut-style file returns the appropriate errors.
    """
    with output_file_params.get("to_dlc_file_expected_exception"):
        file_fixture = output_file_params.get("file_fixture")
        val = request.getfixturevalue(file_fixture)
        file_path = val.get("file_path") if isinstance(val, dict) else val
        save_poses.to_dlc_file(valid_poses_dataset, file_path)


@pytest.mark.parametrize(
    "invalid_poses_dataset, expected_exception",
    invalid_poses_datasets_and_exceptions,
)
def test_to_dlc_file_invalid_dataset(
    invalid_poses_dataset, expected_exception, tmp_path, request
):
    """Test that saving an invalid pose dataset to a valid
    DeepLabCut-style file returns the appropriate errors.
    """
    with pytest.raises(expected_exception):
        save_poses.to_dlc_file(
            request.getfixturevalue(invalid_poses_dataset),
            tmp_path / "test.h5",
            split_individuals=False,
        )


@pytest.mark.parametrize(
    "valid_poses_dataset, split_value",
    [("single_individual_array", True), ("multi_individual_array", False)],
    indirect=["valid_poses_dataset"],
)
def test_auto_split_individuals(valid_poses_dataset, split_value):
    """Test that setting 'split_individuals' to 'auto' yields True
    for single-individual datasets and False for multi-individual ones.
    """
    assert (
        save_poses._auto_split_individuals(valid_poses_dataset) == split_value
    )


@pytest.mark.parametrize(
    "valid_poses_dataset, split_individuals",
    [
        ("single_individual_array", True),  # single-individual, split
        ("multi_individual_array", False),  # multi-individual, no split
        ("single_individual_array", False),  # single-individual, no split
        ("multi_individual_array", True),  # multi-individual, split
    ],
    indirect=["valid_poses_dataset"],
)
def test_to_dlc_style_df_split_individuals(
    valid_poses_dataset, split_individuals
):
    """Test that the `split_individuals` argument affects the behaviour
    of the `to_dlc_style_df` function as expected.
    """
    df = save_poses.to_dlc_style_df(valid_poses_dataset, split_individuals)
    # Get the names of the individuals in the dataset
    ind_names = valid_poses_dataset.individuals.values
    if split_individuals is False:
        # this should produce a single df in multi-animal DLC format
        assert isinstance(df, pd.DataFrame)
        assert df.columns.names == [
            "scorer",
            "individuals",
            "bodyparts",
            "coords",
        ]
        assert all(
            [ind in df.columns.get_level_values("individuals")]
            for ind in ind_names
        )
    elif split_individuals is True:
        # this should produce a dict of dfs in single-animal DLC format
        assert isinstance(df, dict)
        for ind in ind_names:
            assert ind in df
            assert isinstance(df[ind], pd.DataFrame)
            assert df[ind].columns.names == [
                "scorer",
                "bodyparts",
                "coords",
            ]


@pytest.mark.parametrize(
    "split_individuals, expected_exception",
    [
        (True, does_not_raise()),
        (False, does_not_raise()),
        ("auto", does_not_raise()),
        ("1", pytest.raises(ValueError, match="boolean or 'auto'")),
    ],
)
def test_to_dlc_file_split_individuals(
    valid_poses_dataset,
    new_h5_file,
    split_individuals,
    expected_exception,
):
    """Test that the `split_individuals` argument affects the behaviour
    of the `to_dlc_file` function as expected.
    """
    with expected_exception:
        save_poses.to_dlc_file(
            valid_poses_dataset, new_h5_file, split_individuals
        )
        # Get the names of the individuals in the dataset
        ind_names = valid_poses_dataset.individuals.values
        # "auto" becomes False, default valid dataset is multi-individual
        if split_individuals in [False, "auto"]:
            # this should save only one file
            assert new_h5_file.is_file()
            new_h5_file.unlink()
        elif split_individuals is True:
            # this should save one file per individual
            for ind in ind_names:
                file_path_ind = Path(f"{new_h5_file.with_suffix('')}_{ind}.h5")
                assert file_path_ind.is_file()
                file_path_ind.unlink()


def test_to_lp_file_valid_dataset(
    output_file_params, valid_poses_dataset, request
):
    """Test that saving a valid pose dataset to a valid/invalid
    LightningPose-style file returns the appropriate errors.
    """
    with output_file_params.get("to_lp_file_expected_exception"):
        file_fixture = output_file_params.get("file_fixture")
        val = request.getfixturevalue(file_fixture)
        file_path = val.get("file_path") if isinstance(val, dict) else val
        save_poses.to_lp_file(valid_poses_dataset, file_path)


@pytest.mark.parametrize(
    "invalid_poses_dataset, expected_exception",
    invalid_poses_datasets_and_exceptions,
)
def test_to_lp_file_invalid_dataset(
    invalid_poses_dataset, expected_exception, tmp_path, request
):
    """Test that saving an invalid pose dataset to a valid
    LightningPose-style file returns the appropriate errors.
    """
    with pytest.raises(expected_exception):
        save_poses.to_lp_file(
            request.getfixturevalue(invalid_poses_dataset),
            tmp_path / "test.csv",
        )


def test_to_sleap_analysis_file_valid_dataset(
    output_file_params, valid_poses_dataset, request
):
    """Test that saving a valid pose dataset to a valid/invalid
    SLEAP-style file returns the appropriate errors.
    """
    with output_file_params.get("to_sleap_file_expected_exception"):
        file_fixture = output_file_params.get("file_fixture")
        val = request.getfixturevalue(file_fixture)
        file_path = val.get("file_path") if isinstance(val, dict) else val
        save_poses.to_sleap_analysis_file(valid_poses_dataset, file_path)


@pytest.mark.parametrize(
    "invalid_poses_dataset, expected_exception",
    invalid_poses_datasets_and_exceptions,
)
def test_to_sleap_analysis_file_invalid_dataset(
    invalid_poses_dataset, expected_exception, new_h5_file, request
):
    """Test that saving an invalid pose dataset to a valid
    SLEAP-style file returns the appropriate errors.
    """
    with pytest.raises(expected_exception):
        save_poses.to_sleap_analysis_file(
            request.getfixturevalue(invalid_poses_dataset),
            new_h5_file,
        )


nwb_file_expectations_ind = {
    "default_kwargs-single_ind": {
        "nwbfile_kwargs": [
            {"session_description": "not set", "identifier": "id_0"}
        ],
        "processing_module_kwargs": [
            {"description": "processed behavioral data"}
        ],
        "subject_kwargs": [{"subject_id": "id_0"}],
        "pose_estimation_kwargs": [
            {"name": "PoseEstimation", "source_software": "test"}
        ],
        "skeleton_kwargs": [
            {"name": "skeleton_id_0", "nodes": ["centroid", "left", "right"]}
        ],
    },
    "default_kwargs-multi_ind": {
        "nwbfile_kwargs": [
            {"session_description": "not set", "identifier": "id_0"},
            {"session_description": "not set", "identifier": "id_1"},
        ],
        "processing_module_kwargs": [
            {"description": "processed behavioral data"},
            {"description": "processed behavioral data"},
        ],
        "subject_kwargs": [
            {"subject_id": "id_0"},
            {"subject_id": "id_1"},
        ],
        "pose_estimation_kwargs": [
            {"name": "PoseEstimation", "source_software": "test"},
            {"name": "PoseEstimation", "source_software": "test"},
        ],
        "skeleton_kwargs": [
            {"name": "skeleton_id_0", "nodes": ["centroid", "left", "right"]},
            {"name": "skeleton_id_1", "nodes": ["centroid", "left", "right"]},
        ],
    },
    "shared_kwargs-single_ind": {
        "nwbfile_kwargs": [
            {"session_description": "test session", "identifier": "subj0"}
        ],
        "processing_module_kwargs": [
            {"description": "processed behav for test session"}
        ],
        "subject_kwargs": [{"age": "P90D", "subject_id": "subj0"}],
        "pose_estimation_kwargs": [
            {"name": "subj0", "source_software": "other"}
        ],
        "skeleton_kwargs": [
            {"name": "skeleton0", "nodes": ["anchor", "left_ear", "right_ear"]}
        ],
    },
    "shared_kwargs-multi_ind": {
        "nwbfile_kwargs": [
            {"session_description": "test session", "identifier": "id_0"},
            {"session_description": "test session", "identifier": "id_1"},
        ],
        "processing_module_kwargs": [
            {"description": "processed behav for test session"},
            {"description": "processed behav for test session"},
        ],
        "subject_kwargs": [
            {"age": "P90D", "subject_id": "id_0"},
            {"age": "P90D", "subject_id": "id_1"},
        ],
        "pose_estimation_kwargs": [
            {"name": "id_0", "source_software": "other"},
            {"name": "id_1", "source_software": "other"},
        ],
        "skeleton_kwargs": [
            {"name": "id_0", "nodes": ["anchor", "left_ear", "right_ear"]},
            {"name": "id_1", "nodes": ["anchor", "left_ear", "right_ear"]},
        ],
    },
    "custom_kwargs-single_ind": {
        "nwbfile_kwargs": [
            {"session_description": "session subj0", "identifier": "subj0"}
        ],
        "processing_module_kwargs": [
            {"description": "processed behav for subj0"}
        ],
        "subject_kwargs": [{"age": "P90D", "subject_id": "subj0"}],
        "pose_estimation_kwargs": [
            {"name": "subj0", "source_software": "other0"}
        ],
        "skeleton_kwargs": [
            {"name": "skeleton_id_0", "nodes": ["node1", "node2", "node3"]}
        ],
    },
    "custom_kwargs-multi_ind": {
        "nwbfile_kwargs": [
            {"session_description": "session subj0", "identifier": "subj0"},
            {"session_description": "session subj1", "identifier": "subj1"},
        ],
        "processing_module_kwargs": [
            {"description": "processed behav for subj0"},
            {"description": "processed behav for subj1"},
        ],
        "subject_kwargs": [
            {"age": "P90D", "subject_id": "subj0"},
            {"age": "P91D", "subject_id": "subj1"},
        ],
        "pose_estimation_kwargs": [
            {"name": "subj0", "source_software": "other0"},
            {"name": "subj1", "source_software": "other1"},
        ],
        "skeleton_kwargs": [
            {"name": "skeleton_id_0", "nodes": ["node1", "node2", "node3"]},
            {"name": "skeleton_id_1", "nodes": ["node4", "node5", "node6"]},
        ],
    },
}


@pytest.mark.parametrize(
    "selection_fn",
    [
        lambda ds: ds.sel(individuals="id_0"),
        lambda ds: ds,
    ],
    ids=["single_ind", "multi_ind"],
)
@pytest.mark.parametrize(
    "config",
    [None, "shared_nwb_config", "per_entity_nwb_config"],
    ids=["default_kwargs", "shared_kwargs", "custom_kwargs"],
)
def test_to_nwb_file_with_single_or_multi_ind_ds(
    selection_fn, config, valid_poses_dataset, request
):
    """Test that saving single-/multi-individual poses dataset to NWBFile(s)
    with various configurations correctly sets the per-individual
    NWBFile, Subject, PoseEstimation, and Skeleton attributes.
    """
    ds = selection_fn(valid_poses_dataset)
    config = request.getfixturevalue(config) if config else config
    test_id = request.node.callspec.id
    nwb_files = save_poses.to_nwb_file(ds, config)
    if ds.individuals.size == 1:
        nwb_files = [nwb_files]
    actual_nwbfile_kwargs = []
    actual_processing_module_kwargs = []
    actual_subject_kwargs = []
    actual_pose_estimation_kwargs = []
    actual_skeleton_kwargs = []
    expected_nwbfile_kwargs = nwb_file_expectations_ind.get(test_id).get(
        "nwbfile_kwargs"
    )
    expected_processing_module_kwargs = nwb_file_expectations_ind.get(
        test_id
    ).get("processing_module_kwargs")
    expected_subject_kwargs = nwb_file_expectations_ind.get(test_id).get(
        "subject_kwargs"
    )
    expected_pose_estimation_kwargs = nwb_file_expectations_ind.get(
        test_id
    ).get("pose_estimation_kwargs")
    expected_skeleton_kwargs = nwb_file_expectations_ind.get(test_id).get(
        "skeleton_kwargs"
    )
    for expected_skeleton, expected_pe, nwb_file in zip(
        expected_skeleton_kwargs,
        expected_pose_estimation_kwargs,
        nwb_files,
        strict=True,
    ):
        processing_module = nwb_file.processing["behavior"]
        pose_estimation = processing_module[expected_pe["name"]]
        skeleton = processing_module["Skeletons"][expected_skeleton["name"]]
        actual_nwbfile_kwargs.append(
            {key: getattr(nwb_file, key) for key in expected_nwbfile_kwargs[0]}
        )
        actual_processing_module_kwargs.append(
            {
                key: getattr(processing_module, key)
                for key in expected_processing_module_kwargs[0]
            }
        )
        actual_subject_kwargs.append(
            {
                key: getattr(nwb_file.subject, key)
                for key in expected_subject_kwargs[0]
            }
        )
        actual_pose_estimation_kwargs.append(
            {
                key: getattr(pose_estimation, key)
                for key in expected_pose_estimation_kwargs[0]
            }
        )
        actual_skeleton_kwargs.append(
            {
                key: getattr(skeleton, key)
                for key in expected_skeleton_kwargs[0]
            }
        )
    assert actual_nwbfile_kwargs == expected_nwbfile_kwargs
    assert actual_processing_module_kwargs == expected_processing_module_kwargs
    assert actual_subject_kwargs == expected_subject_kwargs
    assert actual_pose_estimation_kwargs == expected_pose_estimation_kwargs
    assert actual_skeleton_kwargs == expected_skeleton_kwargs


nwb_file_expectations_keypoint = {
    "default_kwargs-single_keypoint": {
        "pose_estimation_series_kwargs": [
            {
                "reference_frame": "(0,0,0) corresponds to ...",
                "unit": "pixels",
                "name": "centroid",
            },
        ],
    },
    "default_kwargs-multi_keypoint": {
        "pose_estimation_series_kwargs": [
            {
                "reference_frame": "(0,0,0) corresponds to ...",
                "unit": "pixels",
                "name": "centroid",
            },
            {
                "reference_frame": "(0,0,0) corresponds to ...",
                "unit": "pixels",
                "name": "left",
            },
            {
                "reference_frame": "(0,0,0) corresponds to ...",
                "unit": "pixels",
                "name": "right",
            },
        ],
    },
    "shared_kwargs-single_keypoint": {
        "pose_estimation_series_kwargs": [
            {"reference_frame": "(0,0) is ...", "name": "anchor"}
        ],
    },
    "shared_kwargs-multi_keypoint": {
        "pose_estimation_series_kwargs": [
            {"reference_frame": "(0,0) is ...", "name": "centroid"},
            {"reference_frame": "(0,0) is ...", "name": "left"},
            {"reference_frame": "(0,0) is ...", "name": "right"},
        ],
    },
    "custom_kwargs-single_keypoint": {
        "pose_estimation_series_kwargs": [{"name": "anchor"}],
    },
    "custom_kwargs-multi_keypoint": {
        "pose_estimation_series_kwargs": [
            {"name": "anchor"},
            {"name": "left_ear"},
            {"name": "right"},
        ],
    },
}


@pytest.mark.parametrize(
    "selection_fn",
    [
        lambda ds: ds.sel(keypoints=["centroid"]),
        lambda ds: ds,
    ],
    ids=["single_keypoint", "multi_keypoint"],
)
@pytest.mark.parametrize(
    "config",
    [None, "shared_nwb_config", "per_entity_nwb_config"],
    ids=["default_kwargs", "shared_kwargs", "custom_kwargs"],
)
def test_to_nwb_file_with_single_or_multi_keypoint_ds(
    selection_fn, config, valid_poses_dataset, request
):
    """Test saving single-/multi-keypoint poses dataset to NWBFile(s) with
    various configurations correctly sets the per-keypoint PoseEstimationSeries
    attributes.
    """
    # Use single-individual dataset for simplicity
    ds = selection_fn(valid_poses_dataset).isel(individuals=0)
    test_id = request.node.callspec.id
    config = request.getfixturevalue(config) if config else config
    nwb_file = save_poses.to_nwb_file(ds, config)
    pose_estimation_name = (
        "PoseEstimation" if "default" in test_id else "subj0"
    )
    expected_pes_kwargs = nwb_file_expectations_keypoint.get(test_id).get(
        "pose_estimation_series_kwargs"
    )
    actual_pes_kwargs = []
    for pes in nwb_file.processing["behavior"][
        pose_estimation_name
    ].pose_estimation_series.values():
        actual_pes_kwargs.append(
            {key: getattr(pes, key) for key in expected_pes_kwargs[0]}
        )
    assert actual_pes_kwargs == expected_pes_kwargs


def test_remove_unoccupied_tracks(valid_poses_dataset):
    """Test that removing unoccupied tracks from a valid pose dataset
    returns the expected result.
    """
    new_individuals = [f"id_{i}" for i in range(3)]
    # Add new individual with NaN data
    ds = valid_poses_dataset.reindex(individuals=new_individuals)
    ds = save_poses._remove_unoccupied_tracks(ds)
    xr.testing.assert_equal(ds, valid_poses_dataset)


# -------------- Motion-BIDS saver tests ----------------------------


def test_to_motion_bids_creates_files(valid_poses_dataset, tmp_path):
    """Test that to_motion_bids creates the expected output files."""
    paths = save_poses.to_motion_bids(valid_poses_dataset, tmp_path)
    assert isinstance(paths, dict)
    assert set(paths.keys()) == {
        "motion_tsv",
        "channels_tsv",
        "motion_json",
    }
    for key, path in paths.items():
        assert path.exists(), f"{key} file was not created"


def test_to_motion_bids_filenames(valid_poses_dataset, tmp_path):
    """Test that the BIDS filenames follow the expected pattern."""
    paths = save_poses.to_motion_bids(
        valid_poses_dataset,
        tmp_path,
        task="walking",
        tracksys="markerless",
        subject="02",
    )
    prefix = "sub-02_task-walking_tracksys-markerless"
    assert paths["motion_tsv"].name == f"{prefix}_motion.tsv"
    assert paths["channels_tsv"].name == f"{prefix}_channels.tsv"
    assert paths["motion_json"].name == f"{prefix}_motion.json"


def test_to_motion_bids_with_session(valid_poses_dataset, tmp_path):
    """Test that session entity appears in the filename when specified."""
    paths = save_poses.to_motion_bids(
        valid_poses_dataset,
        tmp_path,
        session="pre",
    )
    assert "ses-pre" in paths["motion_tsv"].name


def test_to_motion_bids_metadata_json(valid_poses_dataset, tmp_path):
    """Test that the Motion-BIDS JSON metadata is correct."""
    import json

    ds = valid_poses_dataset.copy()
    ds.attrs["fps"] = 25

    paths = save_poses.to_motion_bids(
        ds,
        tmp_path,
        task="tracking",
        tracksys="pose",
    )

    with open(paths["motion_json"]) as f:
        metadata = json.load(f)

    assert "SamplingFrequency" in metadata
    assert metadata["TaskName"] == "tracking"
    assert metadata["TrackingSystemName"] == "pose"
    assert "POSChannelCount" in metadata
    assert "TrackedPointsCount" in metadata


def test_to_motion_bids_metadata_kwargs(valid_poses_dataset, tmp_path):
    """Test that extra metadata_kwargs are written to the JSON sidecar."""
    import json

    paths = save_poses.to_motion_bids(
        valid_poses_dataset,
        tmp_path,
        Manufacturer="OptiTrack",
        Description="Custom experiment",
    )

    with open(paths["motion_json"]) as f:
        metadata = json.load(f)

    assert metadata["Manufacturer"] == "OptiTrack"
    assert metadata["Description"] == "Custom experiment"


def test_to_motion_bids_channels_tsv(valid_poses_dataset, tmp_path):
    """Test the channels TSV has required columns and POS type."""
    paths = save_poses.to_motion_bids(valid_poses_dataset, tmp_path)
    channels_df = pd.read_csv(paths["channels_tsv"], sep="\t")
    required_cols = {"name", "component", "type", "tracked_point", "units"}
    assert required_cols.issubset(set(channels_df.columns))
    assert (channels_df["type"] == "POS").all()


def test_to_motion_bids_motion_tsv_no_header(valid_poses_dataset, tmp_path):
    """Test that the motion TSV has no header row."""
    paths = save_poses.to_motion_bids(valid_poses_dataset, tmp_path)
    with open(paths["motion_tsv"]) as f:
        first_line = f.readline().strip()
    # All values should be numeric (no header)
    values = first_line.split("\t")
    for v in values:
        float(v)  # Should not raise


def test_to_motion_bids_multi_individual(valid_poses_dataset, tmp_path):
    """Test that multi-individual datasets include individual column."""
    # valid_poses_dataset defaults to multi_individual_array (2 individuals)
    paths = save_poses.to_motion_bids(valid_poses_dataset, tmp_path)
    channels_df = pd.read_csv(paths["channels_tsv"], sep="\t")
    assert "individual" in channels_df.columns
    individuals = channels_df["individual"].unique().tolist()
    assert len(individuals) == 2


@pytest.mark.parametrize(
    "valid_poses_dataset",
    ["single_individual_array"],
    indirect=True,
)
def test_to_motion_bids_single_individual(valid_poses_dataset, tmp_path):
    """Test that single-individual datasets omit individual column."""
    paths = save_poses.to_motion_bids(valid_poses_dataset, tmp_path)
    channels_df = pd.read_csv(paths["channels_tsv"], sep="\t")
    assert "individual" not in channels_df.columns


def test_to_motion_bids_round_trip(valid_motion_bids_2d):
    """Test that saving then loading preserves position data."""
    ds_loaded = load_poses.from_motion_bids_file(valid_motion_bids_2d)
    output_dir = valid_motion_bids_2d.parent / "roundtrip"
    paths = save_poses.to_motion_bids(
        ds_loaded,
        output_dir,
        task="tracking",
        tracksys="pose",
        subject="01",
    )
    ds_reloaded = load_poses.from_motion_bids_file(paths["motion_tsv"])
    xr.testing.assert_allclose(ds_loaded.position, ds_reloaded.position)
    assert list(ds_loaded.coords["keypoints"].values) == list(
        ds_reloaded.coords["keypoints"].values
    )


def test_to_motion_bids_creates_output_dir(valid_poses_dataset, tmp_path):
    """Test that output directory is created if it does not exist."""
    output_dir = tmp_path / "new_nested" / "dir"
    paths = save_poses.to_motion_bids(valid_poses_dataset, output_dir)
    assert output_dir.exists()
    assert paths["motion_tsv"].exists()
