from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr

import movement.kinematics as kinematics
from movement.io import load_bboxes, load_poses


@pytest.mark.parametrize(
    "deprecated_function, mocked_inputs, patch_context, check_in_message",
    [
        (
            kinematics.compute_displacement,
            {"data": MagicMock(dims=["time", "space"])},
            nullcontext(),
            ["compute_forward_displacement", "compute_backward_displacement"],
        ),
        (
            load_poses.from_file,
            {"file": "dummy_path", "source_software": "SLEAP"},
            patch(
                "movement.io.load_poses.from_sleap_file",
                return_value=xr.Dataset(),
            ),
            ["load_dataset"],
        ),
        (
            load_poses.from_multiview_files,
            {
                "file_dict": {"dummy_view": "dummy_path"},
                "source_software": "SLEAP",
            },
            patch(
                "movement.io.load_poses.from_sleap_file",
                return_value=xr.Dataset(),
            ),
            ["load_multiview_dataset"],
        ),
        (
            load_bboxes.from_file,
            {"file": "dummy_path", "source_software": "VIA-tracks"},
            patch(
                "movement.io.load_bboxes.from_via_tracks_file",
                return_value=xr.Dataset(),
            ),
            ["load_dataset"],
        ),
    ],
)
def test_deprecation(
    deprecated_function, mocked_inputs, patch_context, check_in_message
):
    """Test that calling a deprecated function raises a DeprecationWarning
    with the expected message.
    """
    with (
        patch_context,
        pytest.warns(DeprecationWarning, match="deprecated") as record,
    ):
        _ = deprecated_function(**mocked_inputs)
    # match= checks the message; assert below verifies the function name
    assert f"{deprecated_function.__name__}` is deprecated" in str(
        record[0].message
    )
    assert all(
        message in str(record[0].message) for message in check_in_message
    )


# ---------------- Backwards compatibility tests ----------------


@pytest.mark.parametrize(
    "valid_dataset",
    ["valid_poses_dataset", "valid_bboxes_dataset"],
)
def test_backwards_compatibility_displacement(valid_dataset, request):
    """Test that compute_displacement produces the same output as
    the negative of compute_backward_displacement.
    """
    position = request.getfixturevalue(valid_dataset).position

    with pytest.warns(DeprecationWarning, match="deprecated"):
        result = kinematics.compute_displacement(position)

    expected_result = -kinematics.compute_backward_displacement(position)
    assert result.equals(expected_result), (
        "compute_displacement should produce the same output as "
        "the negative of compute_backward_displacement"
    )
