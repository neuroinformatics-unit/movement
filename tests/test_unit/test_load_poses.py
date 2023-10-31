import numpy as np
import pytest
import xarray as xr
from pytest import POSE_DATA
from sleap_io.io.slp import read_labels

from movement.io import PosesAccessor, load_poses


class TestLoadPoses:
    """Test suite for the load_poses module."""

    @pytest.fixture(scope="module")
    def sleap_user_labels(self):
        """Return a SLEAP `Labels` object containing only
        user-labeled instances.
        """
        slp_file = POSE_DATA.get(
            "SLEAP_three-mice_Aeon_proofread.predictions.slp"
        )
        return read_labels(slp_file)

    @pytest.fixture(scope="module")
    def sleap_predicted_labels(self):
        """Return a SLEAP `Labels` object containing only
        predicted instances.
        """
        slp_file = POSE_DATA.get("SLEAP_single-mouse_EPM.predictions.slp")
        return read_labels(slp_file)

    def assert_dataset(
        self, dataset, file_path=None, expected_source_software=None
    ):
        """Assert that the dataset is a proper xarray Dataset."""
        assert isinstance(dataset, xr.Dataset)
        # Expected variables are present and of right shape/type
        for var in ["pose_tracks", "confidence"]:
            assert var in dataset.data_vars
            assert isinstance(dataset[var], xr.DataArray)
        assert dataset.pose_tracks.ndim == 4
        assert dataset.confidence.shape == dataset.pose_tracks.shape[:-1]
        # Check the dims and coords
        DIM_NAMES = PosesAccessor.dim_names
        assert all([i in dataset.dims for i in DIM_NAMES])
        for d, dim in enumerate(DIM_NAMES[1:]):
            assert dataset.dims[dim] == dataset.pose_tracks.shape[d + 1]
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

    def test_load_from_slp_file(self, sleap_file):
        """Test that loading pose tracks from valid SLEAP files
        returns a proper Dataset."""
        ds = load_poses.from_sleap_file(sleap_file)
        self.assert_dataset(ds, sleap_file, "SLEAP")

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
        ],
    )
    def test_load_from_sleap_slp_file_or_h5_file_returns_same(
        self, slp_file, h5_file
    ):
        """Test that loading pose tracks from SLEAP .slp and .h5 files
        return the same Dataset."""
        slp_file_path = POSE_DATA.get(slp_file)
        h5_file_path = POSE_DATA.get(h5_file)
        ds_from_slp = load_poses.from_sleap_file(slp_file_path)
        ds_from_h5 = load_poses.from_sleap_file(h5_file_path)
        xr.testing.assert_allclose(ds_from_h5, ds_from_slp)

    @pytest.mark.parametrize(
        "sleap_labels, expected",
        [
            ("sleap_user_labels", True),
            ("sleap_predicted_labels", False),
        ],
    )
    def test_sleap_labels_contains_user_labels_only(
        self, sleap_labels, expected, request
    ):
        labels = request.getfixturevalue(sleap_labels)
        assert (
            load_poses._sleap_labels_contains_user_labels_only(labels)
            == expected
        )

    @pytest.mark.parametrize(
        "sleap_labels",
        [
            "sleap_user_labels",
            "sleap_predicted_labels",
        ],
    )
    def test_sleap_user_labels_to_numpy_confidence_equals_one(
        self, sleap_labels, request
    ):
        labels = request.getfixturevalue(sleap_labels)
        confidence = load_poses._sleap_user_labels_to_numpy(labels)[:, :, :, 2]
        mask = np.isnan(confidence)
        assert np.all(confidence[~mask] == 1)

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
        returns a proper Dataset."""
        file_path = POSE_DATA.get(file_name)
        ds = load_poses.from_dlc_file(file_path)
        self.assert_dataset(ds, file_path, "DeepLabCut")

    def test_load_from_dlc_df(self, dlc_style_df):
        """Test that loading pose tracks from a valid DLC-style DataFrame
        returns a proper Dataset."""
        ds = load_poses.from_dlc_df(dlc_style_df)
        self.assert_dataset(ds)

    def test_load_from_dlc_file_csv_or_h5_file_returns_same(self):
        """Test that loading pose tracks from DLC .csv and .h5 files
        return the same Dataset."""
        csv_file_path = POSE_DATA.get("DLC_single-wasp.predictions.csv")
        h5_file_path = POSE_DATA.get("DLC_single-wasp.predictions.h5")
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
            POSE_DATA.get("SLEAP_three-mice_Aeon_proofread.analysis.h5"),
            fps=fps,
        )
        assert ds.time_unit == expected_time_unit
        if expected_fps is None:
            assert ds.fps is expected_fps
        else:
            assert ds.fps == expected_fps
            np.testing.assert_allclose(
                ds.coords["time"].data,
                np.arange(ds.dims["time"], dtype=int) / ds.attrs["fps"],
            )
