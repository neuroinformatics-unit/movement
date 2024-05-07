import numpy as np
import pandas as pd
import pytest

from movement.napari.convert import (
    _replace_nans_with_zeros,
    ds_to_napari_tracks,
)


class TestNapariConvert:
    """Test suite for the movement.napari.convert module."""

    @pytest.fixture
    def confidence_with_some_nan(self, valid_pose_dataset):
        """Return a valid pose dataset with some NaNs in confidence values."""
        valid_pose_dataset["confidence"].loc[
            {"individuals": "ind1", "time": [3, 7, 8]}
        ] = np.nan
        return valid_pose_dataset

    @pytest.fixture
    def confidence_with_all_nan(self, valid_pose_dataset):
        """Return a valid pose dataset with all NaNs in confidence values."""
        valid_pose_dataset["confidence"].data = np.full_like(
            valid_pose_dataset["confidence"].data, np.nan
        )
        return valid_pose_dataset

    datasets_to_test = [
        "valid_pose_dataset",
        "valid_pose_dataset_with_nan",
        "confidence_with_some_nan",
        "confidence_with_all_nan",
    ]

    @pytest.mark.parametrize("ds_name", datasets_to_test)
    def test_replace_confidence_nans_with_zeros(
        self, ds_name, request, caplog
    ):
        """Test that NaN confidence values are replaced with zeros.
        Also checks that a warning is logged when NaNs are found.
        """
        ds = request.getfixturevalue(ds_name)
        nan_coords = (
            ds["confidence"].where(ds["confidence"].isnull(), drop=True).coords
        )

        ds_new = _replace_nans_with_zeros(ds, ["confidence"])

        # Warning should be logged if confidence contained NaNs
        if ds_name in ["confidence_with_some_nan", "confidence_with_all_nan"]:
            assert "NaNs found in confidence" in caplog.records[0].message
            assert caplog.records[0].levelname == "WARNING"
        else:
            assert len(caplog.records) == 0

        # Check that the NaN values have been replaced with zeros
        assert np.all(ds_new["confidence"].notnull())
        assert np.all(ds_new["confidence"].sel(nan_coords).data == 0)

    @pytest.mark.parametrize("ds_name", datasets_to_test)
    def test_ds_to_napari_tracks(self, ds_name, request):
        """Test that the conversion from xarray dataset to napari tracks
        returns the expected data and properties.
        """
        ds = request.getfixturevalue(ds_name)
        n_frames = ds.sizes["time"]
        n_individuals = ds.sizes["individuals"]
        n_keypoints = ds.sizes["keypoints"]
        n_tracks = n_individuals * n_keypoints

        data, props = ds_to_napari_tracks(ds)

        # Check the types and shapes of the returned values
        assert isinstance(data, np.ndarray)
        assert data.shape == (n_frames * n_tracks, 4)
        assert isinstance(props, pd.DataFrame)
        assert len(props) == n_frames * n_tracks

        # Check that the data array has the expected values
        # first column should be each track id repeated for n_frames
        expected_track_ids = np.repeat(np.arange(n_tracks), n_frames)
        np.testing.assert_allclose(data[:, 0], expected_track_ids)
        # 2nd column should be each frame ids repeated for n_tracks
        expected_frame_ids = np.tile(np.arange(n_frames), n_tracks)
        np.testing.assert_allclose(data[:, 1], expected_frame_ids)
        # The last two columns should be the y and x coordinates
        base = np.arange(n_frames, dtype=float)
        expected_yx = np.column_stack(
            (np.tile(base * 4, n_tracks), np.tile(base * base, n_tracks))
        )
        if ds_name == "valid_pose_dataset_with_nan":
            expected_yx[[3, 7, 8, 13, 17, 18]] = np.nan
        np.testing.assert_allclose(data[:, 2:], expected_yx, equal_nan=True)

        # Check that the properties dataframe has the expected values
        expected_columns = ["individual", "keypoint", "time", "confidence"]
        assert all(col in props.columns for col in expected_columns)

        # check that the properties match their expected values
        # note: track_id should be individual_id * n_keypoints + keypoint_id
        expected_individuals = [
            f"ind{i}" for i in expected_track_ids // n_keypoints + 1
        ]
        assert np.array_equal(props["individual"], expected_individuals)
        expected_keypoints = [
            f"key{i}" for i in expected_track_ids % n_keypoints + 1
        ]
        assert np.array_equal(props["keypoint"], expected_keypoints)
        np.testing.assert_allclose(props["time"], expected_frame_ids)

        # Make sure that confidence values are finite
        assert np.all(np.isfinite(props["confidence"]))
        # Check that confidence values are all 1.0
        # Except where NaNs were replaced with zeros
        expected_confidence = np.full(n_frames * n_tracks, 1.0)
        if ds_name == "confidence_with_all_nan":
            expected_confidence = np.zeros(n_frames * n_tracks)
        elif ds_name == "confidence_with_some_nan":
            expected_confidence[[3, 7, 8, 13, 17, 18]] = 0.0
        np.testing.assert_allclose(props["confidence"], expected_confidence)
