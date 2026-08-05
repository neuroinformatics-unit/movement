from contextlib import nullcontext as does_not_raise

import pytest

from movement.io import save


class TestSaveDataset:
    """Tests for save_dataset."""

    def test_dispatches_to_writer(self, valid_poses_dataset, mocker, tmp_path):
        """Test save_dataset forwards to the registered writer, passing the
        dataset, file path and extra kwargs through unchanged.
        """
        mock_writer = mocker.MagicMock()
        mocker.patch.dict(
            "movement.io.save._WRITER_REGISTRY",
            {"StubSoftware": mock_writer},
        )
        file_path = tmp_path / "output"
        save.save_dataset(
            valid_poses_dataset,
            file_path,
            target_software="StubSoftware",
            foo="bar",
        )
        mock_writer.assert_called_once_with(
            valid_poses_dataset, file_path, foo="bar"
        )

    def test_invalid_target_software(self, valid_poses_dataset):
        """Test save_dataset raises an error for invalid target software."""
        with pytest.raises(ValueError, match="Unsupported target_software"):
            save.save_dataset(
                valid_poses_dataset, "some_file", target_software="bogus"
            )

    @pytest.mark.parametrize(
        "target_software, dataset_fixture, mutate_attrs, expected_context",
        [
            (
                "DeepLabCut",
                "valid_bboxes_dataset",
                None,
                pytest.raises(ValueError, match="Missing required"),
            ),
            (
                "VIA-tracks",
                "valid_poses_dataset",
                None,
                pytest.raises(ValueError, match="Missing required"),
            ),
            (
                None,
                "not_a_dataset",
                None,
                pytest.raises(TypeError, match="Expected an xarray Dataset"),
            ),
            (
                "netCDF",
                "valid_bboxes_dataset",
                lambda attrs: attrs.pop("ds_type", None),
                pytest.raises(
                    ValueError, match="Cannot save to 'netCDF'.*ds_type"
                ),
            ),
            (
                "netCDF",
                "valid_bboxes_dataset",
                lambda attrs: attrs.update(ds_type="bogus"),
                pytest.raises(
                    ValueError, match="Cannot save to 'netCDF'.*ds_type"
                ),
            ),
        ],
        ids=[
            "DeepLabCut: rejects bboxes dataset",
            "VIA-tracks: rejects poses dataset",
            "netCDF: rejects non-xarray Dataset",
            "netCDF: rejects xarray Dataset missing ds_type attr",
            "netCDF: rejects xarray Dataset with invalid ds_type attr",
        ],
    )
    def test_rejects_mismatched_ds_type(
        self,
        target_software,
        dataset_fixture,
        mutate_attrs,
        expected_context,
        request,
    ):
        """Test _validate_ds_type called by save_dataset raises the
        appropriate error when the dataset is incompatible with the
        target software.
        """
        ds = request.getfixturevalue(dataset_fixture)
        if mutate_attrs is not None:
            mutate_attrs(ds.attrs)
        with expected_context:
            save.save_dataset(ds, "some_file", target_software=target_software)

    @pytest.mark.parametrize("target_software", [None, "netCDF"])
    @pytest.mark.parametrize(
        "file, expected_context",
        [
            ("dataset.nc", does_not_raise()),
            (
                "dataset.ext1",
                pytest.raises(ValueError, match="Expected file with suffix"),
            ),
        ],
    )
    def test_save_netcdf(
        self,
        valid_poses_dataset,
        target_software,
        file,
        expected_context,
        tmp_path,
    ):
        """Test saving to netCDF (default/explicit) with valid and invalid
        file suffixes. Indirectly tests _to_netcdf_file.
        """
        file_path = tmp_path / file
        with expected_context:
            save.save_dataset(valid_poses_dataset, file_path, target_software)

    @pytest.mark.parametrize(
        "valid_poses_dataset, identifiers",
        [
            ("single_individual_array", None),
            ("multi_individual_array", ["subj0", "subj"]),
        ],
        ids=[
            "single individual: writes to given path verbatim",
            "multi individual: appends identifier to path",
        ],
        indirect=["valid_poses_dataset"],
    )
    def test_save_nwb(
        self, valid_poses_dataset, identifiers, mocker, tmp_path
    ):
        """Test saving to NWB writes a single file for a single-individual
        dataset, or one file per individual (identifier appended to the file
        path) for a multi-individual dataset.
        """
        import pynwb

        file_path = tmp_path / "out.nwb"
        if identifiers is None:
            to_nwb_file_return = mocker.MagicMock(spec=pynwb.file.NWBFile)
            expected_paths = [file_path]
        else:
            to_nwb_file_return = [
                mocker.MagicMock(identifier=identifier)
                for identifier in identifiers
            ]
            expected_paths = [
                tmp_path / f"out_{identifier}.nwb"
                for identifier in identifiers
            ]
        mocker.patch(
            "movement.io.save_poses.to_nwb_file",
            return_value=to_nwb_file_return,
        )
        mock_write = mocker.patch("movement.io.save_poses._write_nwb_to_disk")
        save.save_dataset(
            valid_poses_dataset, file_path, target_software="NWB"
        )
        written_paths = [call.args[1] for call in mock_write.call_args_list]
        assert written_paths == expected_paths


class TestRegisterWriterDecorator:
    """Tests for the register_writer decorator."""

    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        """Patch both writer registries and provide a reusable stub writer."""
        mocker.patch.dict("movement.io.save._WRITER_REGISTRY")
        mocker.patch.dict("movement.io.save._WRITER_DS_TYPE_REGISTRY")
        self.mock_writer = mocker.MagicMock()

    def _register(self, **kwargs):
        """Register a stub writer function with the decorator."""
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
            (None, "some_file.ext1", does_not_raise()),
            ({".ext1"}, "some_file.ext1", does_not_raise()),
            (
                {".ext1"},
                "some_file.ext2",
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
