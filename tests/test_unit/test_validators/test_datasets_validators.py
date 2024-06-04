from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from movement.io.validators.datasets import ValidPosesDataset


@pytest.mark.parametrize(
    "invalid_position_array, log_message",
    [
        (
            None,
            f"Expected a numpy array, but got {type(None)}.",
        ),  # invalid, argument is non-optional
        (
            [1, 2, 3],
            f"Expected a numpy array, but got {type(list())}.",
        ),  # not an ndarray
        (
            np.zeros((10, 2, 3)),
            "Expected `position_array` to have 4 dimensions, but got 3.",
        ),  # not 4d
        (
            np.zeros((10, 2, 3, 4)),
            "Expected `position_array` to have 2 or 3 "
            "spatial dimensions, but got 4.",
        ),  # last dim not 2 or 3
    ],
)
def test_poses_dataset_validator_with_invalid_position_array(
    invalid_position_array, log_message
):
    """Test that invalid position arrays raise the appropriate errors."""
    with pytest.raises(ValueError) as excinfo:
        ValidPosesDataset(position_array=invalid_position_array)
    assert str(excinfo.value) == log_message


@pytest.mark.parametrize(
    "confidence_array, expected_exception",
    [
        (
            np.ones((10, 3, 2)),
            pytest.raises(ValueError),
        ),  # will not match position_array shape
        (
            [1, 2, 3],
            pytest.raises(ValueError),
        ),  # not an ndarray, should raise ValueError
        (
            None,
            does_not_raise(),
        ),  # valid, should default to array of NaNs
    ],
)
def test_poses_dataset_validator_confidence_array(
    confidence_array,
    expected_exception,
    valid_position_array,
):
    """Test that invalid confidence arrays raise the appropriate errors."""
    with expected_exception:
        poses = ValidPosesDataset(
            position_array=valid_position_array("multi_individual_array"),
            confidence_array=confidence_array,
        )
        if confidence_array is None:
            assert np.all(np.isnan(poses.confidence_array))


def test_poses_dataset_validator_keypoint_names(
    position_array_params, valid_position_array
):
    """Test that invalid keypoint names raise the appropriate errors."""
    with position_array_params.get("keypoint_names_expected_exception") as e:
        poses = ValidPosesDataset(
            position_array=valid_position_array(
                position_array_params.get("array_type")
            ),
            keypoint_names=position_array_params.get("names"),
        )
        assert poses.keypoint_names == e


def test_poses_dataset_validator_individual_names(
    position_array_params, valid_position_array
):
    """Test that invalid keypoint names raise the appropriate errors."""
    with position_array_params.get("individual_names_expected_exception") as e:
        poses = ValidPosesDataset(
            position_array=valid_position_array(
                position_array_params.get("array_type")
            ),
            individual_names=position_array_params.get("names"),
        )
        assert poses.individual_names == e


@pytest.mark.parametrize(
    "source_software, expected_exception",
    [
        (None, does_not_raise()),
        ("SLEAP", does_not_raise()),
        ("DeepLabCut", does_not_raise()),
        ("LightningPose", pytest.raises(ValueError)),
        ("fake_software", does_not_raise()),
        (5, pytest.raises(TypeError)),  # not a string
    ],
)
def test_poses_dataset_validator_source_software(
    valid_position_array, source_software, expected_exception
):
    """Test that the source_software attribute is validated properly.
    LightnigPose is incompatible with multi-individual arrays.
    """
    with expected_exception:
        ds = ValidPosesDataset(
            position_array=valid_position_array("multi_individual_array"),
            source_software=source_software,
        )

        if source_software is not None:
            assert ds.source_software == source_software
        else:
            assert ds.source_software is None
