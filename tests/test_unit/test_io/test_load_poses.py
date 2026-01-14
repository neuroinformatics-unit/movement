"""Test suite for the load_poses module."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from pytest import DATA_PATHS

from movement.io import load_poses
from movement.validators.datasets import ValidPosesInputs

expected_values_poses = {
    "vars_dims": {"position": 4, "confidence": 3},
    "dim_names": ValidPosesInputs.DIM_NAMES,
}


def test_load_from_sleap_file(sleap_file, helpers):
    """Test that loading pose tracks from valid SLEAP files
    returns a proper Dataset.
    """
    ds = load_poses.from_sleap_file(sleap_file)
    expected_values = {
        **expected_values_poses,
        "source_software": "SLEAP",
        "file_path": sleap_file,
    }
    helpers.assert_valid_dataset(ds, expected_values)


def test_load_from_sleap_file_without_tracks(sleap_file_without_tracks):
    """Test that loading pose tracks from valid SLEAP files
    with tracks removed returns a dataset that matches the
    original file, except for the individual names which are
    set to default.
    """
    ds_from_trackless = load_poses.from_sleap_file(sleap_file_without_tracks)
    ds_from_tracked = load_poses.from_sleap_file(
        DATA_PATHS.get("SLEAP_single-mouse_EPM.analysis.h5")
    )
    # Check if the "individual" coordinate matches
    # the assigned default "id_0"
    assert ds_from_trackless.individual == ["id_0"]
    xr.testing.assert_allclose(
        ds_from_trackless.drop_vars("individual"),
        ds_from_tracked.drop_vars("individual"),
    )


@pytest.mark.parametrize(
    "slp_file, h5_file",
    [
        (
            "SLEAP_single-mouse_EPM.analysis.h5",
            "SLEAP_single-mouse_EPM.predictions.slp",
        ),
        (
            "SLEAP_three-mice_Aeon_proofread.analysis.h5",
            "SLEAP_three-mice_Aeon_proofread.predictions.slp",
        ),
        (
            "SLEAP_three-mice_Aeon_mixed-labels.analysis.h5",
            "SLEAP_three-mice_Aeon_mixed-labels.predictions.slp",
        ),
    ],
)
def test_load_from_sleap_slp_file_or_h5_file_returns_same(slp_file, h5_file):
    """Test that loading pose tracks from SLEAP .slp and .h5 files
    return the same Dataset.
    """
    slp_file_path = DATA_PATHS.get(slp_file)
    h5_file_path = DATA_PATHS.get(h5_file)
    ds_from_slp = load_poses.from_sleap_file(slp_file_path)
    ds_from_h5 = load_poses.from_sleap_file(h5_file_path)
    xr.testing.assert_allclose(ds_from_h5, ds_from_slp)


@pytest.mark.parametrize(
    "file_name",
    [
        "DLC_single-wasp.predictions.h5",
        "DLC_single-wasp.predictions.csv",
        "DLC_two-mice.predictions.csv",
    ],
)
def test_load_from_dlc_file(file_name, helpers):
    """Test that loading pose tracks from valid DLC files
    returns a proper Dataset.
    """
    file_path = DATA_PATHS.get(file_name)
    ds = load_poses.from_dlc_file(file_path)
    expected_values = {
        **expected_values_poses,
        "source_software": "DeepLabCut",
        "file_path": file_path,
    }
    helpers.assert_valid_dataset(ds, expected_values)


@pytest.mark.parametrize(
    "poses_df_fixture, source_software",
    [
        ("valid_dlc_poses_df", "DeepLabCut"),
        ("valid_dlc_poses_df", "LightningPose"),
        ("valid_dlc_poses_df", None),
        ("valid_dlc_3d_poses_df", "DeepLabCut"),
    ],
)
def test_load_from_dlc_style_df(
    poses_df_fixture, source_software, helpers, request
):
    """Test loading pose tracks from DLC-style DataFrames (2D and 3D)."""
    df = request.getfixturevalue(poses_df_fixture)
    ds = load_poses.from_dlc_style_df(df, source_software=source_software)
    expected_values = {
        **expected_values_poses,
        "source_software": source_software,
    }
    helpers.assert_valid_dataset(ds, expected_values)


def test_load_from_dlc_file_csv_or_h5_file_returns_same():
    """Test that loading pose tracks from DLC .csv and .h5 files
    return the same Dataset.
    """
    csv_file_path = DATA_PATHS.get("DLC_single-wasp.predictions.csv")
    h5_file_path = DATA_PATHS.get("DLC_single-wasp.predictions.h5")
    ds_from_csv = load_poses.from_dlc_file(csv_file_path)
    ds_from_h5 = load_poses.from_dlc_file(h5_file_path)
    xr.testing.assert_allclose(ds_from_h5, ds_from_csv)


@pytest.mark.filterwarnings(
    "ignore:.*Setting fps to None.:UserWarning",
)
@pytest.mark.parametrize(
    "fps, expected_fps, expected_time_unit",
    [
        (None, None, "frames"),
        (-5, None, "frames"),
        (0, None, "frames"),
        (30, 30, "seconds"),
        (60.0, 60, "seconds"),
    ],
)
def test_fps_and_time_coords(fps, expected_fps, expected_time_unit):
    """Test that time coordinates are set according to the provided fps."""
    ds = load_poses.from_sleap_file(
        DATA_PATHS.get("SLEAP_three-mice_Aeon_proofread.analysis.h5"),
        fps=fps,
    )
    assert ds.time_unit == expected_time_unit
    if expected_fps is None:
        assert "fps" not in ds.attrs
    else:
        assert ds.fps == expected_fps
        np.testing.assert_allclose(
            ds.coords["time"].data,
            np.arange(ds.sizes["time"], dtype=int) / ds.attrs["fps"],
        )


@pytest.mark.parametrize(
    "file_name",
    [
        "LP_mouse-face_AIND.predictions.csv",
        "LP_mouse-twoview_AIND.predictions.csv",
    ],
)
def test_load_from_lp_file(file_name, helpers):
    """Test that loading pose tracks from valid LightningPose (LP) files
    returns a proper Dataset.
    """
    file_path = DATA_PATHS.get(file_name)
    ds = load_poses.from_lp_file(file_path)
    expected_values = {
        **expected_values_poses,
        "source_software": "LightningPose",
        "file_path": file_path,
    }
    helpers.assert_valid_dataset(ds, expected_values)


def test_load_from_lp_or_dlc_file_returns_same():
    """Test that loading a single-animal DeepLabCut-style .csv file
    using either the `from_lp_file` or `from_dlc_file` function
    returns the same Dataset (except for the source_software).
    """
    file_path = DATA_PATHS.get("LP_mouse-face_AIND.predictions.csv")
    ds_drom_lp = load_poses.from_lp_file(file_path)
    ds_from_dlc = load_poses.from_dlc_file(file_path)
    xr.testing.assert_allclose(ds_from_dlc, ds_drom_lp)
    assert ds_drom_lp.source_software == "LightningPose"
    assert ds_from_dlc.source_software == "DeepLabCut"


def test_load_multi_individual_from_lp_file_raises():
    """Test that loading a multi-individual .csv file using the
    `from_lp_file` function raises a ValueError.
    """
    file_path = DATA_PATHS.get("DLC_two-mice.predictions.csv")
    with pytest.raises(ValueError, match="only supports single-individual"):
        load_poses.from_lp_file(file_path)


@pytest.mark.parametrize(
    "source_software",
    ["DeepLabCut", "SLEAP", "LightningPose", "Anipose", "NWB", "Unknown"],
)
@pytest.mark.parametrize("fps", [None, 30, 60.0])
def test_from_file_delegates_correctly(source_software, fps, caplog):
    """Test that the from_file() function delegates to the correct
    loader function according to the source_software.
    """
    software_to_loader = {
        "DeepLabCut": "movement.io.load_poses.from_dlc_file",
        "SLEAP": "movement.io.load_poses.from_sleap_file",
        "LightningPose": "movement.io.load_poses.from_lp_file",
        "Anipose": "movement.io.load_poses.from_anipose_file",
        "NWB": "movement.io.load_poses.from_nwb_file",
    }
    if source_software == "Unknown":
        with pytest.raises(ValueError, match="Unsupported source"):
            load_poses.from_file("some_file", source_software)
    else:
        with patch(software_to_loader[source_software]) as mock_loader:
            load_poses.from_file("some_file", source_software, fps)
            expected_call_args = (
                ("some_file", fps)
                if source_software != "NWB"
                else ("some_file",)
            )
            mock_loader.assert_called_with(*expected_call_args)
            if source_software == "NWB" and fps is not None:
                assert "fps argument is ignored" in caplog.messages[0]


@pytest.mark.parametrize("source_software", [None, "SLEAP"])
def test_from_numpy_valid(valid_poses_arrays, source_software, helpers):
    """Test that loading pose tracks from a multi-animal numpy array
    with valid parameters returns a proper Dataset.
    """
    poses_arrays = valid_poses_arrays("multi_individual_array")
    ds = load_poses.from_numpy(
        poses_arrays["position"],
        poses_arrays["confidence"],
        individual_names=["id_0", "id_1"],
        keypoint_names=["centroid", "left", "right"],
        fps=None,
        source_software=source_software,
    )
    expected_values = {
        **expected_values_poses,
        "source_software": source_software,
    }
    helpers.assert_valid_dataset(ds, expected_values)


def test_from_multiview_files():
    """Test loading pose tracks from multiple files (representing
    different views).
    """
    view_names = ["view_0", "view_1"]
    file_path_dict = {
        view: DATA_PATHS.get("DLC_single-wasp.predictions.h5")
        for view in view_names
    }
    multi_view_ds = load_poses.from_multiview_files(
        file_path_dict, source_software="DeepLabCut"
    )
    assert isinstance(multi_view_ds, xr.Dataset)
    assert "view" in multi_view_ds.dims
    assert multi_view_ds.view.values.tolist() == view_names


def test_load_from_anipose_file():
    """Test that loading pose tracks from an Anipose triangulation
    csv file returns the same Dataset.
    """
    file_path = DATA_PATHS.get(
        "anipose_mouse-paw_anipose-paper.triangulation.csv"
    )
    ds = load_poses.from_anipose_file(file_path)
    assert ds.position.shape == (246, 3, 6, 1)
    assert ds.confidence.shape == (246, 6, 1)
    assert ds.coords["keypoint"].values.tolist() == [
        "l-base",
        "l-edge",
        "l-middle",
        "r-base",
        "r-edge",
        "r-middle",
    ]


@pytest.mark.parametrize("kwargs", [{}, {"rate": 10.0, "starting_time": 0.0}])
@pytest.mark.parametrize("input_type", ["nwb_file", "nwbfile_object"])
def test_load_from_nwb_file(input_type, kwargs, request):
    """Test loading poses from an NWB file path or NWBFile object.
    ``kwargs`` determine whether the PoseEstimationSeries in the NWB file
    are created with default ``timestamps`` (empty kwargs) or without
    timestamps, by providing ``rate`` and ``starting_time``.
    """
    nwb_file = request.getfixturevalue(input_type)(**kwargs)
    ds_from_file_path = load_poses.from_nwb_file(nwb_file)
    assert ds_from_file_path.sizes == {
        "time": 100,
        "individual": 1,
        "keypoint": 3,
        "space": 2,
    }
    expected_attrs = {
        "ds_type": "poses",
        "fps": 10.0,
        "time_unit": "seconds",
        "source_software": "DeepLabCut",
    }
    if input_type == "nwb_file":
        expected_attrs["source_file"] = nwb_file
    assert ds_from_file_path.attrs == expected_attrs
