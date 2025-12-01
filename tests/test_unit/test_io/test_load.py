import pytest

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
        mocker.patch.dict(load._REGISTRY, {source_software: mock_loader})
        load.from_file("some_file", source_software, fps)
        expected_call_args = (
            ("some_file", fps) if source_software != "NWB" else ("some_file",)
        )
        mock_loader.assert_called_with(*expected_call_args)
        if source_software == "NWB" and fps is not None:
            assert "fps argument is ignored" in caplog.messages[0]
