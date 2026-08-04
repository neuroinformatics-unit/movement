"""Test the unified ``save_dataset`` entry point."""

from contextlib import nullcontext as does_not_raise

import pytest
import xarray as xr

from movement.io import save


@pytest.mark.parametrize("target_software", [None, "netCDF"])
def test_save_dataset_netcdf_roundtrip(
    valid_poses_dataset, target_software, tmp_path
):
    """A dataset saved to netCDF (default target) loads back unchanged."""
    file_path = tmp_path / "dataset.nc"
    save.save_dataset(valid_poses_dataset, file_path, target_software)
    assert file_path.is_file()
    loaded = xr.load_dataset(file_path)
    xr.testing.assert_allclose(loaded, valid_poses_dataset)
    assert loaded.attrs == valid_poses_dataset.attrs


def test_save_dataset_netcdf_requires_nc_suffix(valid_poses_dataset, tmp_path):
    """Saving to netCDF with a non-.nc suffix raises an error."""
    with pytest.raises(ValueError, match="Expected file with suffix"):
        save.save_dataset(valid_poses_dataset, tmp_path / "dataset.csv")


def test_save_dataset_rejects_non_dataset(tmp_path):
    """Passing a non-Dataset object raises a TypeError."""
    with pytest.raises(TypeError, match="Expected an xarray Dataset"):
        save.save_dataset([1, 2, 3], tmp_path / "dataset.nc")


def test_save_dataset_invalid_target_software(valid_poses_dataset, tmp_path):
    """An unsupported target raises a helpful ValueError."""
    with pytest.raises(ValueError, match="Unsupported target_software"):
        save.save_dataset(
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
    save.save_dataset(
        ds, file_path, target_software=target_software, foo="bar"
    )
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
        save.save_dataset(
            ds, tmp_path / "output", target_software=target_software
        )


def test_save_dataset_netcdf_rejects_missing_ds_type(tmp_path):
    """Saving an xr.Dataset with no ``ds_type`` attr to netCDF now errors,
    instead of silently writing an unvalidated dataset to disk.
    """
    ds = xr.Dataset({"position": (["time"], [1.0, 2.0])})
    with pytest.raises(ValueError, match="Cannot save to 'netCDF'"):
        save.save_dataset(ds, tmp_path / "dataset.nc")


def test_save_dataset_netcdf_rejects_invalid_ds_type(tmp_path):
    """Saving an xr.Dataset with an unrecognized ``ds_type`` attr errors."""
    ds = xr.Dataset({"position": (["time"], [1.0, 2.0])})
    ds.attrs["ds_type"] = "not-a-real-type"
    with pytest.raises(ValueError, match="Cannot save to 'netCDF'"):
        save.save_dataset(ds, tmp_path / "dataset.nc")


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
    save.save_dataset(single, file_path, target_software="NWB")
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
    save.save_dataset(valid_poses_dataset, file_path, target_software="NWB")
    written_paths = [call.args[1] for call in mock_write.call_args_list]
    assert written_paths == [
        tmp_path / "out_id_0.nwb",
        tmp_path / "out_id_1.nwb",
    ]


class TestRegisterWriter:
    """Tests for the ``register_writer`` decorator."""

    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        """Patch both writer registries and provide a reusable stub writer."""
        mocker.patch.dict("movement.io.save._WRITER_REGISTRY")
        mocker.patch.dict("movement.io.save._WRITER_DS_TYPE_REGISTRY")
        self.mock_writer = mocker.MagicMock()

    def _register(self, **kwargs):
        return save.register_writer("StubSoftware", **kwargs)(self.mock_writer)

    @pytest.mark.parametrize(
        "ds_type, dataset_fixture",
        [
            ("poses", "valid_poses_dataset"),
            ("bboxes", "valid_bboxes_dataset"),
            (None, "valid_poses_dataset"),
            (None, "valid_bboxes_dataset"),
        ],
        ids=[
            "poses-specific: uses ValidPosesInputs",
            "bboxes-specific: uses ValidBboxesInputs",
            "type-agnostic: uses ValidPosesInputs based on ds_type attr",
            "type-agnostic: uses ValidBboxesInputs based on ds_type attr",
        ],
    )
    @pytest.mark.parametrize(
        "allowed_suffixes, file_path, expected_context",
        [
            (None, "dummy_path.ext1", does_not_raise()),
            ({".ext1"}, "dummy_path.ext1", does_not_raise()),
            (
                {".ext1"},
                "dummy_path.ext2",
                pytest.raises(ValueError, match="Expected file with suffix"),
            ),
        ],
    )
    def test_validates_ds_type_and_suffix(
        self,
        dataset_fixture,
        ds_type,
        allowed_suffixes,
        file_path,
        expected_context,
        request,
    ):
        """Test the decorator validates input dataset and file path before
        calling the decorated writer function.
        """
        to_stubsoftware_file = self._register(
            ds_type=ds_type, suffixes=allowed_suffixes
        )
        ds = request.getfixturevalue(dataset_fixture)
        with expected_context:
            to_stubsoftware_file(ds, file_path)

    def test_rejects_mismatched_ds_type(self, valid_bboxes_dataset):
        """Test the decorator rejects a dataset of the wrong ds_type."""
        to_stubsoftware_file = self._register(ds_type="poses")
        with pytest.raises(ValueError, match="Missing required"):
            to_stubsoftware_file(valid_bboxes_dataset, "out.ext1")
        self.mock_writer.assert_not_called()

    def test_forwards_path_and_kwargs(self, valid_poses_dataset, tmp_path):
        """Test the decorator forwards a validated Path and any extra
        kwargs through to the writer.
        """
        to_stubsoftware_file = self._register(ds_type="poses")
        file_path = tmp_path / "out.ext1"
        to_stubsoftware_file(valid_poses_dataset, str(file_path), foo="bar")
        called_ds, called_path = self.mock_writer.call_args.args
        assert called_ds is valid_poses_dataset
        assert called_path == file_path
        assert self.mock_writer.call_args.kwargs == {"foo": "bar"}

    @pytest.mark.parametrize("ds_type", ["poses", "bboxes", None])
    def test_populates_registries(self, ds_type):
        """Test the wrapper and ds_type are recorded in both registries."""
        to_stubsoftware_file = self._register(ds_type=ds_type)
        assert save._WRITER_REGISTRY["StubSoftware"] is to_stubsoftware_file
        assert save._WRITER_DS_TYPE_REGISTRY["StubSoftware"] == ds_type
