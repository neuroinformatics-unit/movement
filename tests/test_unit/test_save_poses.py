from contextlib import nullcontext as does_not_raise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import DATA_PATHS

from movement.io import load_poses, nwb, save_poses

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


@pytest.mark.parametrize(
    "selection_fn",
    [lambda ds: ds.drop_sel(individuals="id_1"), lambda ds: ds],
    ids=["single_nwb_file", "multiple_nwb_files"],
)
def test_to_nwb_file_valid_input(
    selection_fn, valid_poses_dataset, initialised_nwb_file_object
):
    ds = selection_fn(valid_poses_dataset)
    nwbfiles = (
        [initialised_nwb_file_object(ind) for ind in ds.individuals.values]
        if ds.individuals.size > 1
        else initialised_nwb_file_object()
    )
    save_poses.to_nwb_file(ds, nwbfiles)
    if not isinstance(nwbfiles, list):
        nwbfiles = [nwbfiles]
    assert all(
        [
            "PoseEstimation" in nwbfile.processing["behavior"].data_interfaces
            for nwbfile in nwbfiles
        ]
    )
    assert all(
        [
            "Skeletons" in nwbfile.processing["behavior"].data_interfaces
            for nwbfile in nwbfiles
        ]
    )


nwb_file_kwargs_expectations = {
    "default_kwargs-single_nwb_file": {
        "session_description": "not set",
        "identifier": ["not set"],
    },
    "default_kwargs-multiple_nwb_files": {
        "session_description": "not set",
        "identifier": ["id_0", "id_1"],
    },
    "shared_kwargs-single_nwb_file": {
        "session_description": "test session",
        "identifier": ["subj0"],
    },
    "shared_kwargs-multiple_nwb_files": {
        "session_description": "test session",
        "identifier": ["id_0", "id_1"],
    },
    "kwargs_per_ind-single_nwb_file": {
        "session_description": "test session",
        "identifier": ["subj0"],
    },
    "kwargs_per_ind-multiple_nwb_files": {
        "session_description": "test session",
        "identifier": ["subj0", "subj1"],
    },
}


@pytest.mark.parametrize(
    "selection_fn",
    [
        lambda ds: ds.drop_sel(individuals="id_1"),
        lambda ds: ds,
    ],
    ids=["single_nwb_file", "multiple_nwb_files"],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        None,
        {"session_description": "test session", "identifier": "subj0"},
        {
            "id_0": {
                "session_description": "test session",
                "identifier": "subj0",
            },
            "id_1": {
                "session_description": "test session",
                "identifier": "subj1",
            },
        },
    ],
    ids=["default_kwargs", "shared_kwargs", "kwargs_per_ind"],
)
def test_to_nwb_file_min_nwbfile_kwargs(
    selection_fn, kwargs, valid_poses_dataset, request
):
    """Test saving single-/multi-individual poses dataset to NWBFile(s)
    with default or custom ``nwbfile_kwargs``.
    """
    ds = selection_fn(valid_poses_dataset)
    test_id = request.node.callspec.id
    config = nwb.NWBFileSaveConfig(nwbfile_kwargs=kwargs)
    if test_id == "kwargs_per_ind-single_nwb_file":
        # error as too many kwargs to choose from
        with pytest.raises(ValueError, match=".*no individual was provided."):
            save_poses.to_nwb_file_min(ds, config)
        # recreate nwbfile_kwargs with only one individual
        config.nwbfile_kwargs = {
            k: v for k, v in config.nwbfile_kwargs.items() if k == "id_0"
        }
    nwb_files = save_poses.to_nwb_file_min(ds, config)
    actual = [
        (file.session_description, file.identifier) for file in nwb_files
    ]
    expected = nwb_file_kwargs_expectations.get(test_id)
    expected = [
        (expected["session_description"], id) for id in expected["identifier"]
    ]
    assert actual == expected


subject_kwargs_expectations = {
    "default_kwargs-single_nwb_file": [
        {"age__reference": "birth", "subject_id": "id_0"}
    ],
    "default_kwargs-multiple_nwb_files": [
        {
            "age__reference": "birth",
            "subject_id": "id_0",
        },
        {
            "age__reference": "birth",
            "subject_id": "id_1",
        },
    ],
    "shared_kwargs-single_nwb_file": [
        {
            "age__reference": "birth",
            "age": "P90D",
            "subject_id": "subj0",
        }
    ],
    "shared_kwargs-multiple_nwb_files": [
        {
            "age__reference": "birth",
            "age": "P90D",
            "subject_id": "id_0",
        },
        {
            "age__reference": "birth",
            "age": "P90D",
            "subject_id": "id_1",
        },
    ],
    "kwargs_per_ind-single_nwb_file": [
        {
            "age__reference": "birth",
            "age": "P90D",
            "subject_id": "subj0",
        }
    ],
    "kwargs_per_ind-multiple_nwb_files": [
        {
            "age__reference": "birth",
            "age": "P90D",
            "subject_id": "subj0",
        },
        {
            "age__reference": "birth",
            "age": "P91D",
            "subject_id": "subj1",
        },
    ],
}


@pytest.mark.parametrize(
    "selection_fn",
    [
        lambda ds: ds.drop_sel(individuals="id_1"),
        lambda ds: ds,
    ],
    ids=["single_nwb_file", "multiple_nwb_files"],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        None,
        {"age": "P90D", "subject_id": "subj0"},
        {
            "id_0": {"age": "P90D", "subject_id": "subj0"},
            "id_1": {"age": "P91D", "subject_id": "subj1"},
        },
    ],
    ids=["default_kwargs", "shared_kwargs", "kwargs_per_ind"],
)
def test_to_nwb_file_subject_kwargs(
    selection_fn, kwargs, valid_poses_dataset, request
):
    """Test saving single-/multi-individual poses dataset to NWBFile(s)
    with default or custom ``subject_kwargs``.
    """
    ds = selection_fn(valid_poses_dataset)
    test_id = request.node.callspec.id
    config = nwb.NWBFileSaveConfig(subject_kwargs=kwargs)
    nwb_files = save_poses.to_nwb_file_min(ds, config)
    actual = [file.subject.fields for file in nwb_files]
    expected = subject_kwargs_expectations.get(test_id)
    assert actual == expected


pose_estimation_series_kwargs_expectations = {
    "default_kwargs-single_keypoint": [
        {
            "reference_frame": "(0,0,0) corresponds to ...",
            "unit": "pixels",
            "name": "centroid",
        },
    ],
    "default_kwargs-multiple_keypoints": [
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
    "shared_kwargs-single_keypoint": [
        {"reference_frame": "(0,0) is ...", "name": "anchor"}
    ],
    "shared_kwargs-multiple_keypoints": [
        {"reference_frame": "(0,0) is ...", "name": "centroid"},
        {"reference_frame": "(0,0) is ...", "name": "left"},
        {"reference_frame": "(0,0) is ...", "name": "right"},
    ],
    "kwargs_per_keypoint-single_keypoint": [
        {"name": "anchor"},
    ],
    "kwargs_per_keypoint-multiple_keypoints": [
        {"name": "anchor"},
        {"name": "left_ear"},
        {"name": "right"},
    ],
}


@pytest.mark.parametrize(
    "selection_fn",
    [
        lambda ds: ds.drop_sel(keypoints=["left", "right"]),
        lambda ds: ds,
    ],
    ids=["single_keypoint", "multiple_keypoints"],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        None,
        {"reference_frame": "(0,0) is ...", "name": "anchor"},
        {
            "centroid": {"name": "anchor"},
            "left": {"name": "left_ear"},
        },
    ],
    ids=["default_kwargs", "shared_kwargs", "kwargs_per_keypoint"],
)
def test_to_nwb_file_pose_estimation_series_kwargs(
    selection_fn, kwargs, valid_poses_dataset, request
):
    """Test saving single-/multi-keypoint poses dataset to NWBFile(s)
    with default or custom ``pose_estimation_series_kwargs``.
    """
    ds = selection_fn(valid_poses_dataset).drop_sel(individuals="id_1")
    test_id = request.node.callspec.id
    config = nwb.NWBFileSaveConfig(pose_estimation_series_kwargs=kwargs)
    nwb_file = save_poses.to_nwb_file_min(ds, config)[0]
    expected = pose_estimation_series_kwargs_expectations.get(test_id)
    expected_keys = expected[0].keys()
    actual = []
    for pes in nwb_file.processing["behavior"][
        "PoseEstimation"
    ].pose_estimation_series.values():
        actual.append({key: getattr(pes, key) for key in expected_keys})
    assert actual == expected


def test_to_nwb_file_invalid_input(
    valid_poses_dataset, initialised_nwb_file_object
):
    with pytest.raises(ValueError):
        save_poses.to_nwb_file(
            valid_poses_dataset, initialised_nwb_file_object()
        )


def test_remove_unoccupied_tracks(valid_poses_dataset):
    """Test that removing unoccupied tracks from a valid pose dataset
    returns the expected result.
    """
    new_individuals = [f"id_{i}" for i in range(3)]
    # Add new individual with NaN data
    ds = valid_poses_dataset.reindex(individuals=new_individuals)
    ds = save_poses._remove_unoccupied_tracks(ds)
    xr.testing.assert_equal(ds, valid_poses_dataset)
