from unittest.mock import patch

import h5py
import numpy as np
import pytest
import xarray as xr
from pytest import DATA_PATHS
from sleap_io.io.slp import read_labels, write_labels
from sleap_io.model.labels import LabeledFrame, Labels

from movement import MovementDataset
from movement.io import load_poses


class TestLoadPoses:
    """Test suite for the load_poses module."""

    @pytest.fixture
    def sleap_slp_file_without_tracks(self, tmp_path):
        """Mock and return the path to a SLEAP .slp file without tracks."""
        sleap_file = DATA_PATHS.get("SLEAP_single-mouse_EPM.predictions.slp")
        labels = read_labels(sleap_file)
        file_path = tmp_path / "track_is_none.slp"
        lfs = []
        for lf in labels.labeled_frames:
            instances = []
            for inst in lf.instances:
                inst.track = None
                inst.tracking_score = 0
                instances.append(inst)
            lfs.append(
                LabeledFrame(
                    video=lf.video, frame_idx=lf.frame_idx, instances=instances
                )
            )
        write_labels(
            file_path,
            Labels(
                labeled_frames=lfs,
                videos=labels.videos,
                skeletons=labels.skeletons,
            ),
        )
        return file_path

    @pytest.fixture
    def sleap_h5_file_without_tracks(self, tmp_path):
        """Mock and return the path to a SLEAP .h5 file without tracks."""
        sleap_file = DATA_PATHS.get("SLEAP_single-mouse_EPM.analysis.h5")
        file_path = tmp_path / "track_is_none.h5"
        with h5py.File(sleap_file, "r") as f1, h5py.File(file_path, "w") as f2:
            for key in list(f1.keys()):
                if key == "track_names":
                    f2.create_dataset(key, data=[])
                else:
                    f1.copy(key, f2, name=key)
        return file_path

    @pytest.fixture(
        params=[
            "sleap_h5_file_without_tracks",
            "sleap_slp_file_without_tracks",
        ]
    )
    def sleap_file_without_tracks(self, request):
        """Fixture to parametrize the SLEAP files without tracks."""
        return request.getfixturevalue(request.param)

    def assert_dataset(
        self, dataset, file_path=None, expected_source_software=None
    ):
        """Assert that the dataset is a proper xarray Dataset."""
        assert isinstance(dataset, xr.Dataset)
        # Expected variables are present and of right shape/type
        for var in ["position", "confidence"]:
            assert var in dataset.data_vars
            assert isinstance(dataset[var], xr.DataArray)
        assert dataset.position.ndim == 4
        assert dataset.confidence.shape == dataset.position.shape[:-1]
        # Check the dims and coords
        DIM_NAMES = MovementDataset.dim_names_per_ds_type["poses"]
        assert all([i in dataset.dims for i in DIM_NAMES])
        for d, dim in enumerate(DIM_NAMES[1:]):
            assert dataset.sizes[dim] == dataset.position.shape[d + 1]
            assert all(
                [isinstance(s, str) for s in dataset.coords[dim].values]
            )
        assert all([i in dataset.coords["space"] for i in ["x", "y"]])
        # Check the metadata attributes
        assert (
            dataset.source_file is None
            if file_path is None
            else dataset.source_file == file_path.as_posix()
        )
        assert (
            dataset.source_software is None
            if expected_source_software is None
            else dataset.source_software == expected_source_software
        )
        assert dataset.fps is None

    def test_load_from_sleap_file(self, sleap_file):
        """Test that loading pose tracks from valid SLEAP files
        returns a proper Dataset.
        """
        ds = load_poses.from_sleap_file(sleap_file)
        self.assert_dataset(ds, sleap_file, "SLEAP")

    def test_load_from_sleap_file_without_tracks(
        self, sleap_file_without_tracks
    ):
        """Test that loading pose tracks from valid SLEAP files
        with tracks removed returns a dataset that matches the
        original file, except for the individual names which are
        set to default.
        """
        ds_from_trackless = load_poses.from_sleap_file(
            sleap_file_without_tracks
        )
        ds_from_tracked = load_poses.from_sleap_file(
            DATA_PATHS.get("SLEAP_single-mouse_EPM.analysis.h5")
        )
        # Check if the "individuals" coordinate matches
        # the assigned default "individuals_0"
        assert ds_from_trackless.individuals == ["individual_0"]
        xr.testing.assert_allclose(
            ds_from_trackless.drop_vars("individuals"),
            ds_from_tracked.drop_vars("individuals"),
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
    def test_load_from_sleap_slp_file_or_h5_file_returns_same(
        self, slp_file, h5_file
    ):
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
    def test_load_from_dlc_file(self, file_name):
        """Test that loading pose tracks from valid DLC files
        returns a proper Dataset.
        """
        file_path = DATA_PATHS.get(file_name)
        ds = load_poses.from_dlc_file(file_path)
        self.assert_dataset(ds, file_path, "DeepLabCut")

    @pytest.mark.parametrize(
        "source_software", ["DeepLabCut", "LightningPose", None]
    )
    def test_load_from_dlc_style_df(self, dlc_style_df, source_software):
        """Test that loading pose tracks from a valid DLC-style DataFrame
        returns a proper Dataset.
        """
        ds = load_poses.from_dlc_style_df(
            dlc_style_df, source_software=source_software
        )
        self.assert_dataset(ds, expected_source_software=source_software)

    def test_load_from_dlc_file_csv_or_h5_file_returns_same(self):
        """Test that loading pose tracks from DLC .csv and .h5 files
        return the same Dataset.
        """
        csv_file_path = DATA_PATHS.get("DLC_single-wasp.predictions.csv")
        h5_file_path = DATA_PATHS.get("DLC_single-wasp.predictions.h5")
        ds_from_csv = load_poses.from_dlc_file(csv_file_path)
        ds_from_h5 = load_poses.from_dlc_file(h5_file_path)
        xr.testing.assert_allclose(ds_from_h5, ds_from_csv)

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
    def test_fps_and_time_coords(self, fps, expected_fps, expected_time_unit):
        """Test that time coordinates are set according to the provided fps."""
        ds = load_poses.from_sleap_file(
            DATA_PATHS.get("SLEAP_three-mice_Aeon_proofread.analysis.h5"),
            fps=fps,
        )
        assert ds.time_unit == expected_time_unit
        if expected_fps is None:
            assert ds.fps is expected_fps
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
    def test_load_from_lp_file(self, file_name):
        """Test that loading pose tracks from valid LightningPose (LP) files
        returns a proper Dataset.
        """
        file_path = DATA_PATHS.get(file_name)
        ds = load_poses.from_lp_file(file_path)
        self.assert_dataset(ds, file_path, "LightningPose")

    def test_load_from_lp_or_dlc_file_returns_same(self):
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

    def test_load_multi_individual_from_lp_file_raises(self):
        """Test that loading a multi-individual .csv file using the
        `from_lp_file` function raises a ValueError.
        """
        file_path = DATA_PATHS.get("DLC_two-mice.predictions.csv")
        with pytest.raises(ValueError):
            load_poses.from_lp_file(file_path)

    @pytest.mark.parametrize(
        "source_software", ["SLEAP", "DeepLabCut", "LightningPose", "Unknown"]
    )
    @pytest.mark.parametrize("fps", [None, 30, 60.0])
    def test_from_file_delegates_correctly(self, source_software, fps):
        """Test that the from_file() function delegates to the correct
        loader function according to the source_software.
        """
        software_to_loader = {
            "SLEAP": "movement.io.load_poses.from_sleap_file",
            "DeepLabCut": "movement.io.load_poses.from_dlc_file",
            "LightningPose": "movement.io.load_poses.from_lp_file",
        }

        if source_software == "Unknown":
            with pytest.raises(ValueError, match="Unsupported source"):
                load_poses.from_file("some_file", source_software)
        else:
            with patch(software_to_loader[source_software]) as mock_loader:
                load_poses.from_file("some_file", source_software, fps)
                mock_loader.assert_called_with("some_file", fps)

    @pytest.mark.parametrize("source_software", [None, "SLEAP"])
    def test_from_numpy_valid(
        self,
        valid_position_array,
        source_software,
    ):
        """Test that loading pose tracks from a multi-animal numpy array
        with valid parameters returns a proper Dataset.
        """
        valid_position = valid_position_array("multi_individual_array")
        rng = np.random.default_rng(seed=42)
        valid_confidence = rng.random(valid_position.shape[:-1])

        ds = load_poses.from_numpy(
            valid_position,
            valid_confidence,
            individual_names=["mouse1", "mouse2"],
            keypoint_names=["snout", "tail"],
            fps=None,
            source_software=source_software,
        )
        self.assert_dataset(ds, expected_source_software=source_software)
