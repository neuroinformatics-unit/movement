from contextlib import nullcontext as does_not_raise
from typing import ClassVar

import pytest
import xarray as xr
from attrs import define, field, validators
from pytest import DATA_PATHS
from requests_cache import Path

from movement.io import load


@define
class StubValidFile:
    """A stub file validator for testing purposes."""

    suffixes: ClassVar[set[str]] = {".stub"}
    file: Path = field(converter=Path)
    loader_arg_to_validate: int = field(default=0, validator=validators.ge(0))


@pytest.mark.parametrize(
    "file_validators, expected_file_type",
    [(None, str), (StubValidFile, StubValidFile)],
)
def test_register_loader_decorator(file_validators, expected_file_type):
    """Test register_loader with and without file validators."""

    @load.register_loader("StubSoftware", file_validators=file_validators)
    def stub_loader_fn(
        file: str, loader_arg_to_validate: int = 0
    ) -> xr.Dataset:
        """Stub loader function for testing."""
        ds = xr.Dataset({"loader_arg": (["x"], [loader_arg_to_validate])})
        ds.attrs["file"] = file
        return ds

    ds = stub_loader_fn("file.stub")
    assert "StubSoftware" in load._LOADER_REGISTRY
    assert load._LOADER_REGISTRY["StubSoftware"] is stub_loader_fn
    assert isinstance(ds.attrs["file"], expected_file_type)


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
    """Test from_file delegates to the correct loader function
    according to the source_software.
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
    "file_fixture, source_software, params, expected_context",
    [
        (
            "via_tracks_csv",
            "VIA-tracks",
            {
                "fps": 30,
                "use_frame_numbers_from_file": True,
                "frame_regexp": r"(0\d*)\.\w+$",
            },
            does_not_raise(),
        ),
        ("sleap_slp_file", "SLEAP", None, does_not_raise()),
        ("dlc_h5_file", "DeepLabCut", None, does_not_raise()),
        ("anipose_csv_file", "Anipose", None, does_not_raise()),
        ("nwbfile_object", "NWB", None, does_not_raise()),
        (
            "dlc_csv_file",
            "SLEAP",
            None,
            pytest.raises(ValueError, match="Unsupported format"),
        ),
    ],
)
def test_from_file(
    file_fixture, source_software, params, expected_context, request
):
    """Test from_file with real files (from various source software)
    and parameters.
    """
    file_path = request.getfixturevalue(file_fixture)
    if file_fixture.startswith("nwb"):
        file_path = file_path()  # NWB fixture is a callable
    with expected_context:
        ds = load.from_file(
            file_path,
            source_software,
            **(params or {}),
        )
        assert isinstance(ds, xr.Dataset)


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


def test_get_validator_kwargs():
    """Test _get_validator_kwargs correctly extracts kwargs
    for file validators.
    """
    loader_kwargs = {"loader_arg_to_validate": 10, "unused_arg": "ignored"}
    validator_kwargs = load._get_validator_kwargs(
        StubValidFile, loader_kwargs=loader_kwargs
    )
    assert validator_kwargs == {"loader_arg_to_validate": 10}


def test_build_suffix_map():
    """Test _build_suffix_map correctly builds a mapping from
    suffixes to validator classes.
    """
    suffix_map = load._build_suffix_map([StubValidFile])
    assert suffix_map == {".stub": StubValidFile}
