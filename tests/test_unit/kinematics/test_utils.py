"""Tests for internal utility functions in kinematics."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from movement.kinematics.utils import (
    _compute_scaled_path_length,
    _warn_about_nan_proportion,
)


@pytest.mark.parametrize(
    "nan_warn_threshold, expected_exception",
    [
        (1, does_not_raise()),
        (0.2, does_not_raise()),
        (-1, pytest.raises(ValueError, match="between 0 and 1")),
    ],
)
def test_path_length_warns_about_nans(
    valid_poses_dataset_with_nan,
    nan_warn_threshold,
    expected_exception,
    caplog,
):
    position = valid_poses_dataset_with_nan.position
    with expected_exception:
        _warn_about_nan_proportion(position, nan_warn_threshold)
        if 0.1 < nan_warn_threshold < 0.5:
            assert caplog.records[0].levelname == "WARNING"
            assert "The result may be unreliable" in caplog.records[0].message
            info_msg = caplog.records[1].message
            assert caplog.records[1].levelname == "INFO"
            assert "Individual: id_0" in info_msg
            assert "Individual: id_1" not in info_msg
            assert "centroid: 3/10 (30.0%)" in info_msg
            assert "right: 10/10 (100.0%)" in info_msg
            assert "left" not in info_msg


def test_compute_scaled_path_length(valid_poses_dataset_with_nan):
    position = valid_poses_dataset_with_nan.position
    path_length = _compute_scaled_path_length(position)
    expected_path_lengths_id_0 = np.array(
        [np.sqrt(2) * 9, np.sqrt(2) * 9, np.nan]
    )
    path_length_id_0 = path_length.sel(individuals="id_0").values
    np.testing.assert_allclose(path_length_id_0, expected_path_lengths_id_0)
