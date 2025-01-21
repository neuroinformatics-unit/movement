import pytest

from movement.utils.reports import report_nan_values


@pytest.mark.parametrize(
    "valid_dataset",
    [
        "valid_poses_dataset",
        "valid_bboxes_dataset",
        "valid_poses_dataset_with_nan",
        "valid_bboxes_dataset_with_nan",
    ],
)
@pytest.mark.parametrize(
    "data_selection, expected_individuals_indices",
    [
        (lambda ds: ds.position, [0, 1]),  # full position data array
        (
            lambda ds: ds.position.isel(individuals=0),
            [0],
        ),  # individual 0 only
    ],
)
def test_report_nan_values_in_position_selecting_individual(
    valid_dataset,
    data_selection,
    expected_individuals_indices,
    request,
):
    """Test that the nan-value reporting function handles position data
    with specific ``individuals``, and that the data array name (position)
    and only the relevant individuals are included in the report.
    """
    # extract relevant position data
    input_dataset = request.getfixturevalue(valid_dataset)
    output_data_array = data_selection(input_dataset)
    # produce report
    report_str = report_nan_values(output_data_array)
    # check report of nan values includes name of data array
    assert output_data_array.name in report_str
    # check report of nan values includes selected individuals only
    list_of_individuals = input_dataset["individuals"].values.tolist()
    all_individuals = set(list_of_individuals)
    expected_individuals = set(
        list_of_individuals[i] for i in expected_individuals_indices
    )
    not_expected_individuals = all_individuals - expected_individuals
    assert all(ind in report_str for ind in expected_individuals) and all(
        ind not in report_str for ind in not_expected_individuals
    ), "Report contains incorrect individuals."


@pytest.mark.parametrize(
    "valid_dataset",
    ["valid_poses_dataset", "valid_poses_dataset_with_nan"],
)
@pytest.mark.parametrize(
    "data_selection, expected_keypoints, expected_individuals",
    [
        (
            lambda ds: ds.position,
            {"centroid", "left", "right"},
            {"id_0", "id_1"},
        ),  # Report nans in position for all keypoints and individuals
        (
            lambda ds: ds.position.sel(keypoints=["centroid", "left"]),
            {"centroid", "left"},
            {"id_0", "id_1"},
        ),  # Report nans in position for 2 keypoints, for all individuals
        (
            lambda ds: ds.position.sel(
                individuals="id_0", keypoints="centroid"
            ),
            set(),
            {"id_0"},
        ),  # Report nans in position for centroid of individual id_0
        # Note: if only 1 keypoint exists, its name is not explicitly reported
    ],
)
def test_report_nan_values_in_position_selecting_keypoint(
    valid_dataset,
    data_selection,
    expected_keypoints,
    expected_individuals,
    request,
):
    """Test that the nan-value reporting function handles position data
    with specific ``keypoints`` , and that the data array name (position)
    and only the relevant keypoints are included in the report.
    """
    # extract relevant position data
    input_dataset = request.getfixturevalue(valid_dataset)
    output_data_array = data_selection(input_dataset)
    # produce report
    report_str = report_nan_values(output_data_array)
    # check report of nan values includes name of data array
    assert output_data_array.name in report_str
    # check report of nan values includes only selected keypoints
    all_keypoints = set(input_dataset["keypoints"].values.tolist())
    not_expected_keypoints = all_keypoints - expected_keypoints
    assert all(kpt in report_str for kpt in expected_keypoints) and all(
        kpt not in report_str for kpt in not_expected_keypoints
    ), "Report contains incorrect keypoints."
    # check report of nan values includes selected individuals only
    all_individuals = set(input_dataset["individuals"].values.tolist())
    not_expected_individuals = all_individuals - expected_individuals
    assert all(ind in report_str for ind in expected_individuals) and all(
        ind not in report_str for ind in not_expected_individuals
    ), "Report contains incorrect individuals."
