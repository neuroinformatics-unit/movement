"""Test suite for Motion-BIDS file validators."""

from contextlib import nullcontext as does_not_raise

import pytest

from movement.validators.files import ValidMotionBidsTSV


@pytest.mark.parametrize(
    "fixture_name, expected_context",
    [
        ("valid_motion_bids_2d", does_not_raise()),
        ("valid_motion_bids_3d", does_not_raise()),
        ("valid_motion_bids_multi_individual", does_not_raise()),
        (
            "motion_bids_missing_channels",
            pytest.raises(
                FileNotFoundError,
                match="Expected companion channels file not found",
            ),
        ),
        (
            "motion_bids_missing_json",
            pytest.raises(
                FileNotFoundError,
                match="Expected companion metadata file not found",
            ),
        ),
        (
            "motion_bids_missing_sampling_freq",
            pytest.raises(
                ValueError,
                match="must contain 'SamplingFrequency'",
            ),
        ),
        (
            "motion_bids_missing_channels_columns",
            pytest.raises(
                ValueError,
                match="missing required columns",
            ),
        ),
        (
            "motion_bids_no_pos_channels",
            pytest.raises(
                ValueError,
                match="must contain at least one channel with type 'POS'",
            ),
        ),
        (
            "motion_bids_with_header",
            pytest.raises(
                ValueError,
                match="must not contain a header row",
            ),
        ),
        (
            "motion_bids_empty_tsv",
            pytest.raises(
                ValueError,
                match="file is empty",
            ),
        ),
        (
            "motion_bids_invalid_json",
            pytest.raises(
                ValueError,
                match="is not a valid JSON file",
            ),
        ),
        (
            "motion_bids_wrong_filename",
            pytest.raises(
                ValueError,
                match="Expected a Motion-BIDS file ending with '_motion.tsv'",
            ),
        ),
        (
            "motion_bids_corrupt_channels",
            pytest.raises(
                ValueError,
                match="Could not parse channels file as TSV",
            ),
        ),
    ],
    ids=[
        "valid 2D Motion-BIDS files",
        "valid 3D Motion-BIDS files",
        "valid multi-individual Motion-BIDS files",
        "missing _channels.tsv companion",
        "missing _motion.json companion",
        "JSON missing SamplingFrequency",
        "channels.tsv missing required columns",
        "channels.tsv has no POS channels",
        "motion.tsv has non-numeric header row",
        "motion.tsv is empty",
        "motion.json is invalid JSON",
        "filename does not end with _motion.tsv",
        "channels.tsv is corrupt/unparsable",
    ],
)
def test_motion_bids_tsv_validator(fixture_name, expected_context, request):
    """Test ValidMotionBidsTSV with valid and invalid inputs."""
    file_path = request.getfixturevalue(fixture_name)
    with expected_context:
        valid = ValidMotionBidsTSV(file_path)
        assert valid.file == file_path
