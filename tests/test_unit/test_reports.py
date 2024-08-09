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
    "data_selection, list_expected_individuals_idcs",
    [
        (lambda ds: ds.position, [0, 1]),
        # Report for the full dataset
        (
            lambda ds: ds.position.isel(individuals=0),
            [0],
        ),  # Report for individual 0 only
    ],
)
def test_report_nan_values_in_position_selecting_individual(
    capsys,
    valid_dataset,
    data_selection,
    list_expected_individuals_idcs,
    request,
):
    """Test that the nan-value reporting function handles position data
    with specific ``individuals`` , and that the dataset name (position)
    and only the relevant individuals are included in the report.
    """
    # apply selection to extract relevant position data
    input_dataset = request.getfixturevalue(valid_dataset)
    output_data_array = data_selection(input_dataset)

    # produce report
    report_str = report_nan_values(output_data_array)

    # check report of nan values includes dataset name
    assert (
        output_data_array.name in report_str
    ), "Dataset name should be in the output"

    # check report of nan values includes selected individuals only
    list_expected_individuals = [
        input_dataset["individuals"][idx].item()
        for idx in list_expected_individuals_idcs
    ]
    list_not_expected_individuals = [
        indiv.item()
        for indiv in input_dataset["individuals"]
        if indiv.item() not in list_expected_individuals
    ]
    assert all([ind in report_str for ind in list_expected_individuals])
    assert all(
        [ind not in report_str for ind in list_not_expected_individuals]
    )


@pytest.mark.parametrize(
    "valid_dataset",
    [
        "valid_poses_dataset",
        "valid_poses_dataset_with_nan",
    ],
)
@pytest.mark.parametrize(
    "data_selection, list_expected_keypoints, list_expected_individuals",
    [
        (
            lambda ds: ds.position,
            ["key1", "key2"],
            ["ind1", "ind2"],
        ),
        # Report nans for all keypoints and individuals
        (
            lambda ds: ds.position.sel(keypoints="key1"),
            [],
            ["ind1", "ind2"],
        ),
        # Report nans for keypoint "key1" only, and all individuals
        # if only one keypoint: not explicitly reported in output
        (
            lambda ds: ds.position.sel(individuals="ind1", keypoints="key1"),
            [],
            ["ind1"],
        ),
        # Report nans for individual "ind1" and keypoint "key1" only
        # if only one keypoint: not explicitly reported in output
    ],
)
def test_report_nan_values_in_position_selecting_keypoint(
    capsys,
    valid_dataset,
    data_selection,
    list_expected_keypoints,
    list_expected_individuals,
    request,
):
    """Test that the nan-value reporting function handles position data
    with specific ``individuals`` , and that the dataset name (position)
    and only the relevant individuals are included in the report.
    """
    # apply selection to extract relevant position data
    input_dataset = request.getfixturevalue(valid_dataset)
    output_data_array = data_selection(input_dataset)

    # produce report
    report_str = report_nan_values(output_data_array)

    # check report of nan values includes dataset name
    assert (
        output_data_array.name in report_str
    ), "Dataset name should be in the output"

    # check report of nan values includes selected kpts only!
    list_not_expected_keypoints = [
        indiv.item()
        for indiv in input_dataset["keypoints"]
        if indiv.item() not in list_expected_keypoints
    ]
    assert all([kpt in report_str for kpt in list_expected_keypoints])
    assert all([kpt not in report_str for kpt in list_not_expected_keypoints])

    # check report of nan values includes selected individuals only
    list_not_expected_individuals = [
        indiv.item()
        for indiv in input_dataset["individuals"]
        if indiv.item() not in list_expected_individuals
    ]
    assert all([ind in report_str for ind in list_expected_individuals])
    assert all(
        [ind not in report_str for ind in list_not_expected_individuals]
    )
