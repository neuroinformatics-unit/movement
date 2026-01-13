import pytest
import xarray as xr
from pytest import DATA_PATHS

from movement.io import load


@pytest.mark.parametrize(
    "source_software, loader_fn",
    [
        ("DeepLabCut", "movement.io.load_poses.from_dlc_file"),
        ("SLEAP", "movement.io.load_poses.from_sleap_file"),
        ("LightningPose", "movement.io.load_poses.from_lp_file"),
        ("Anipose", "movement.io.load_poses.from_anipose_file"),
        ("NWB", "movement.io.load_poses.from_nwb_file"),
        ("VIA-tracks", "movement.io.load_bboxes.from_via_tracks_file"),
        ("Unknown", None),
    ],
)
@pytest.mark.parametrize("fps", [None, 30, 60.0])
def test_from_file_delegates_correctly(
    source_software, loader_fn, fps, caplog, mocker
):
    """Test that the from_file() function delegates to the correct
    loader function according to the source_software.
    """
    if source_software == "Unknown":
        with pytest.raises(ValueError, match="Unsupported source"):
            load.from_file("some_file", source_software)
    else:
        mock_loader = mocker.patch(loader_fn)
        mocker.patch.dict(
            load._LOADER_REGISTRY, {source_software: mock_loader}
        )
        load.from_file("some_file", source_software, fps)
        expected_call_args = (
            ("some_file", fps) if source_software != "NWB" else ("some_file",)
        )
        mock_loader.assert_called_with(*expected_call_args)
        if source_software == "NWB" and fps is not None:
            assert "fps argument is ignored" in caplog.messages[0]


@pytest.mark.parametrize(
    "dataset_name, source_software",
    [
        ("DLC_single-wasp.predictions.h5", "DeepLabCut"),
        ("VIA_multiple-crabs_5-frames_labels.csv", "VIA-tracks"),
    ],
    ids=["Poses", "Bboxes"],
)
def test_from_multiview_files(dataset_name, source_software):
    """Test loading data from multiple files (representing
    different views).
    """
    view_names = ["view_0", "view_1"]
    file_path_dict = {
        view: DATA_PATHS.get(dataset_name) for view in view_names
    }
    multi_view_ds = load.from_multiview_files(
        file_path_dict, source_software=source_software
    )
    assert isinstance(multi_view_ds, xr.Dataset)
    assert "view" in multi_view_ds.dims
    assert multi_view_ds.view.values.tolist() == view_names


@pytest.mark.parametrize("source_software", ["Unknown", "VIA-tracks"])
@pytest.mark.parametrize("fps", [None, 30, 60.0])
@pytest.mark.parametrize("use_frame_numbers_from_file", [True, False])
@pytest.mark.parametrize("frame_regexp", [None, r"frame_(\d+)"])
def test_from_file_bboxes(
    source_software, fps, use_frame_numbers_from_file, frame_regexp, mocker
):
    """Test that the from_file() function delegates to the correct
    loader function according to the source_software.
    """
    software_to_loader = {
        "VIA-tracks": "movement.io.load_bboxes.from_via_tracks_file",
    }
    if source_software == "Unknown":
        with pytest.raises(ValueError, match="Unsupported source"):
            load.from_file(
                "some_file",
                source_software,
                fps,
                use_frame_numbers_from_file=use_frame_numbers_from_file,
                frame_regexp=frame_regexp,
            )
    else:
        mock_loader = mocker.patch(software_to_loader[source_software])
        mocker.patch.dict(
            load._LOADER_REGISTRY, {source_software: mock_loader}
        )
        load.from_file(
            "some_file",
            source_software,
            fps,
            use_frame_numbers_from_file=use_frame_numbers_from_file,
            frame_regexp=frame_regexp,
        )
        mock_loader.assert_called_with(
            "some_file",
            fps,
            use_frame_numbers_from_file=use_frame_numbers_from_file,
            frame_regexp=frame_regexp,
        )
