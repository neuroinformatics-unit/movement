from unittest.mock import MagicMock

import pytest

import movement.kinematics as kinematics


@pytest.mark.parametrize(
    "deprecated_function, mocked_inputs, check_in_message",
    [
        (
            kinematics.compute_displacement,
            {"data": MagicMock(dims=["time", "space"])},
            ["compute_forward_displacement", "compute_backward_displacement"],
        ),
    ],
)
def test_deprecation(deprecated_function, mocked_inputs, check_in_message):
    """Test that calling median_filter raises a DeprecationWarning.
    And that it forwards to rolling_filter with statistic='median'.
    """
    with pytest.warns(DeprecationWarning) as record:
        _ = deprecated_function(**mocked_inputs)

    assert len(record) == 1
    assert isinstance(record[0].message, DeprecationWarning)
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

    with pytest.warns(DeprecationWarning):
        result = kinematics.compute_displacement(position)

    expected_result = -kinematics.compute_backward_displacement(position)
    assert result.equals(expected_result), (
        "compute_displacement should produce the same output as "
        "the negative of compute_backward_displacement"
    )
