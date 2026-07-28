"""Test the unified ``save_dataset`` entry point."""

import pytest
import xarray as xr

from movement.io import save_dataset


@pytest.mark.parametrize("target_software", [None, "netCDF"])
def test_save_dataset_netcdf_roundtrip(
    valid_poses_dataset, target_software, tmp_path
):
    """A dataset saved to netCDF (default target) loads back unchanged."""
    file_path = tmp_path / "dataset.nc"
    save_dataset(valid_poses_dataset, file_path, target_software)
    assert file_path.is_file()
    loaded = xr.load_dataset(file_path)
    xr.testing.assert_allclose(loaded, valid_poses_dataset)
    assert loaded.attrs == valid_poses_dataset.attrs


def test_save_dataset_netcdf_requires_nc_suffix(valid_poses_dataset, tmp_path):
    """Saving to netCDF with a non-.nc suffix raises an error."""
    with pytest.raises(ValueError, match="Expected file with suffix"):
        save_dataset(valid_poses_dataset, tmp_path / "dataset.csv")


def test_save_dataset_rejects_non_dataset(tmp_path):
    """Passing a non-Dataset object raises a TypeError."""
    with pytest.raises(TypeError, match="Expected an xarray Dataset"):
        save_dataset([1, 2, 3], tmp_path / "dataset.nc")


def test_save_dataset_invalid_target_software(valid_poses_dataset, tmp_path):
    """An unsupported target raises a helpful ValueError."""
    with pytest.raises(ValueError, match="Unsupported target_software"):
        save_dataset(
            valid_poses_dataset, tmp_path / "f.txt", target_software="bogus"
        )


@pytest.mark.parametrize(
    "target_software, dataset_fixture",
    [
        ("DeepLabCut", "valid_poses_dataset"),
        ("SLEAP", "valid_poses_dataset"),
        ("LightningPose", "valid_poses_dataset"),
        ("VIA-tracks", "valid_bboxes_dataset"),
    ],
)
def test_save_dataset_dispatches_to_writer(
    target_software, dataset_fixture, request, mocker, tmp_path
):
    """``save_dataset`` forwards to the correct format-specific writer,
    passing the dataset, file path and extra kwargs through unchanged.
    """
    ds = request.getfixturevalue(dataset_fixture)
    mock_writer = mocker.MagicMock()
    mocker.patch.dict(
        "movement.io.save._WRITER_REGISTRY",
        {target_software: mock_writer},
    )
    file_path = tmp_path / "output"
    save_dataset(ds, file_path, target_software=target_software, foo="bar")
    mock_writer.assert_called_once_with(ds, file_path, foo="bar")


@pytest.mark.parametrize(
    "target_software, dataset_fixture",
    [
        ("DeepLabCut", "valid_bboxes_dataset"),  # poses-only target
        ("VIA-tracks", "valid_poses_dataset"),  # bboxes-only target
    ],
)
def test_save_dataset_ds_type_mismatch(
    target_software, dataset_fixture, request, tmp_path
):
    """Saving a dataset to a format meant for the other ds_type errors out."""
    ds = request.getfixturevalue(dataset_fixture)
    with pytest.raises(ValueError, match="Missing required"):
        save_dataset(ds, tmp_path / "output", target_software=target_software)


def test_save_dataset_netcdf_rejects_missing_ds_type(tmp_path):
    """Saving an xr.Dataset with no ``ds_type`` attr to netCDF now errors,
    instead of silently writing an unvalidated dataset to disk.
    """
    ds = xr.Dataset({"position": (["time"], [1.0, 2.0])})
    with pytest.raises(ValueError, match="Cannot save to 'netCDF'"):
        save_dataset(ds, tmp_path / "dataset.nc")


def test_save_dataset_netcdf_rejects_invalid_ds_type(tmp_path):
    """Saving an xr.Dataset with an unrecognized ``ds_type`` attr errors."""
    ds = xr.Dataset({"position": (["time"], [1.0, 2.0])})
    ds.attrs["ds_type"] = "not-a-real-type"
    with pytest.raises(ValueError, match="Cannot save to 'netCDF'"):
        save_dataset(ds, tmp_path / "dataset.nc")


def test_save_dataset_nwb_single_individual(
    valid_poses_dataset, mocker, tmp_path
):
    """A single-individual NWB save writes to the given path verbatim."""
    import pynwb

    single = valid_poses_dataset.isel(individual=[0])
    fake_nwb = mocker.MagicMock(spec=pynwb.file.NWBFile)
    mocker.patch("movement.io.save_poses.to_nwb_file", return_value=fake_nwb)
    mock_write = mocker.patch("movement.io.save_poses._write_nwb_to_disk")
    file_path = tmp_path / "out.nwb"
    save_dataset(single, file_path, target_software="NWB")
    mock_write.assert_called_once_with(fake_nwb, file_path)


def test_save_dataset_nwb_multi_individual(
    valid_poses_dataset, mocker, tmp_path
):
    """A multi-individual NWB save writes one file per individual, with the
    individual name appended to the file path.
    """
    fake_files = [
        mocker.MagicMock(identifier="id_0"),
        mocker.MagicMock(identifier="id_1"),
    ]
    mocker.patch("movement.io.save_poses.to_nwb_file", return_value=fake_files)
    mock_write = mocker.patch("movement.io.save_poses._write_nwb_to_disk")
    file_path = tmp_path / "out.nwb"
    save_dataset(valid_poses_dataset, file_path, target_software="NWB")
    written_paths = [call.args[1] for call in mock_write.call_args_list]
    assert written_paths == [
        tmp_path / "out_id_0.nwb",
        tmp_path / "out_id_1.nwb",
    ]


def test_register_writer_decorator():
    """Test register_writer populates both registries correctly."""
    from movement.io import save

    @save.register_writer("StubSoftware", ds_type="poses")
    def stub_writer_fn(ds, file_path, **kwargs) -> None:
        """Stub writer function for testing."""

    assert save._WRITER_REGISTRY["StubSoftware"] is stub_writer_fn
    assert save._WRITER_DS_TYPE_REGISTRY["StubSoftware"] == "poses"
