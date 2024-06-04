from contextlib import nullcontext as does_not_raise

import pytest

position_arrays = [
    {
        "names": None,
        "array_type": "multi_individual_array",
        "individual_names_expected_exception": does_not_raise(
            ["individual_0", "individual_1"]
        ),
        "keypoint_names_expected_exception": does_not_raise(
            ["keypoint_0", "keypoint_1"]
        ),
    },  # valid input, will generate default names
    {
        "names": ["a", "b"],
        "array_type": "multi_individual_array",
        "individual_names_expected_exception": does_not_raise(["a", "b"]),
        "keypoint_names_expected_exception": does_not_raise(["a", "b"]),
    },  # valid input
    {
        "names": ("a", "b"),
        "array_type": "multi_individual_array",
        "individual_names_expected_exception": does_not_raise(["a", "b"]),
        "keypoint_names_expected_exception": does_not_raise(["a", "b"]),
    },  # valid input, will be converted to ["a", "b"]
    {
        "names": [1, 2],
        "array_type": "multi_individual_array",
        "individual_names_expected_exception": does_not_raise(["1", "2"]),
        "keypoint_names_expected_exception": does_not_raise(["1", "2"]),
    },  # valid input, will be converted to ["1", "2"]
    {
        "names": "a",
        "array_type": "single_individual_array",
        "individual_names_expected_exception": does_not_raise(["a"]),
        "keypoint_names_expected_exception": pytest.raises(ValueError),
    },  # single individual array with multiple keypoints
    {
        "names": "a",
        "array_type": "single_keypoint_array",
        "individual_names_expected_exception": pytest.raises(ValueError),
        "keypoint_names_expected_exception": does_not_raise(["a"]),
    },  # single keypoint array with multiple individuals
    {
        "names": 5,
        "array_type": "multi_individual_array",
        "individual_names_expected_exception": pytest.raises(ValueError),
        "keypoint_names_expected_exception": pytest.raises(ValueError),
    },  # invalid input
]


@pytest.fixture(params=position_arrays)
def position_array_params(request):
    """Return a dictionary containing parameters for testing
    position array keypoint and individual names.
    """
    return request.param
