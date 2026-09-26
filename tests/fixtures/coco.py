"""COCO keypoint results and annotations fixtures."""

import json

import pytest

COCO_KEYPOINT_RESULT_1 = {
    "image_id": 10,
    "category_id": 1,
    "keypoints": [10, 20, 2, 30, 40, 2],
    "score": 0.9,
}

COCO_KEYPOINT_RESULT_2 = {
    "image_id": 10,
    "category_id": 2,
    "keypoints": [50, 60, 2, 70, 80, 2],
    "score": 0.8,
}


@pytest.fixture
def coco_keypoint_results_file(tmp_path):
    """Return a factory that writes a list of COCO results to a JSON file."""

    def _coco_results_file(results):
        file_path = tmp_path / "coco_results.json"

        with open(file_path, "w") as f:
            json.dump(results, f)

        return file_path

    return _coco_results_file


@pytest.fixture
def coco_keypoint_annotations_file(tmp_path):
    """Return a factory that writes a COCO keypoint annotations file with
    the given images, annotations, and categories to a JSON file.
    """

    def _coco_annotations_file(
        categories,
        images=None,
        annotations=None,
    ):
        file_path = tmp_path / "coco_annotations.json"

        with open(file_path, "w") as f:
            json.dump(
                {
                    "images": [] if images is None else images,
                    "annotations": [] if annotations is None else annotations,
                    "categories": categories,
                },
                f,
            )

        return file_path

    return _coco_annotations_file


@pytest.fixture
def coco_keypoint_results_file_valid(coco_keypoint_results_file):
    """Return a valid COCO keypoint results file."""
    results = [
        COCO_KEYPOINT_RESULT_1,
        COCO_KEYPOINT_RESULT_2,
        {
            **COCO_KEYPOINT_RESULT_1,
            "image_id": 20,
            "keypoints": [15, 25, 2, 35, 45, 2],
            "score": 0.7,
        },
    ]
    return coco_keypoint_results_file(results)


@pytest.fixture
def coco_keypoint_results_file_category_as_track(coco_keypoint_results_file):
    """Return a COCO keypoint results file with two detections in one
    frame, listed out of category ID order, so positional and
    category-as-track assignment give different individual orders.
    """
    results = [
        COCO_KEYPOINT_RESULT_2,
        COCO_KEYPOINT_RESULT_1,
    ]
    return coco_keypoint_results_file(results)


@pytest.fixture
def coco_keypoint_annotations_file_category_as_track(
    coco_keypoint_annotations_file,
):
    """Return COCO annotations for category-as-track tests."""
    categories = [
        {
            "id": 1,
            "name": "person",
            "keypoints": ["nose", "left_eye"],
        },
        {
            "id": 2,
            "name": "cat",
            "keypoints": ["nose", "left_eye"],
        },
        {
            "id": 3,
            "name": "dog",
            "keypoints": ["nose", "left_eye"],
        },
    ]
    return coco_keypoint_annotations_file(categories)


@pytest.fixture
def coco_keypoint_results_file_duplicate_category(coco_keypoint_results_file):
    """Return COCO keypoint results with duplicate category in a frame."""
    results = [
        COCO_KEYPOINT_RESULT_1,
        {
            **COCO_KEYPOINT_RESULT_1,
            "keypoints": [50, 60, 2, 70, 80, 2],
            "score": 0.8,
        },
    ]
    return coco_keypoint_results_file(results)


@pytest.fixture
def coco_keypoint_results_file_unknown_category(coco_keypoint_results_file):
    """Return COCO keypoint results with an unknown category."""
    results = [
        {
            **COCO_KEYPOINT_RESULT_1,
            "category_id": 4,
        },
    ]
    return coco_keypoint_results_file(results)


@pytest.fixture
def coco_keypoint_results_file_single_detection(coco_keypoint_results_file):
    """Return COCO keypoint results with a single detection."""
    return coco_keypoint_results_file([COCO_KEYPOINT_RESULT_1])


@pytest.fixture
def coco_keypoint_annotations_file_single_detection(
    coco_keypoint_annotations_file,
):
    """Return COCO annotations containing a person category."""
    categories = [
        {
            "id": 1,
            "name": "person",
            "keypoints": ["nose", "left_eye"],
        },
    ]
    return coco_keypoint_annotations_file(categories)


@pytest.fixture
def coco_keypoint_annotations_file_different_skeleton(
    coco_keypoint_annotations_file,
):
    """Return COCO keypoint annotations with different keypoint skeletons."""
    categories = [
        {
            "id": 1,
            "name": "person",
            "keypoints": ["nose", "left_eye"],
        },
        {
            "id": 2,
            "name": "cat",
            "keypoints": ["nose", "head"],
        },
    ]
    return coco_keypoint_annotations_file(categories)


@pytest.fixture
def coco_keypoint_results_file_without_annotations(coco_keypoint_results_file):
    """Return COCO keypoint results without annotations."""
    results = [
        {
            **COCO_KEYPOINT_RESULT_1,
            "image_id": 20,
            "category_id": 5,
        },
    ]
    return coco_keypoint_results_file(results)


@pytest.fixture
def coco_keypoint_results_file_keypoints_not_divisible_by_3(
    coco_keypoint_results_file,
):
    """Return COCO results with a keypoints list not divisible by 3."""
    results = [
        {
            **COCO_KEYPOINT_RESULT_1,
            "keypoints": [10, 20, 2, 30],
        }
    ]
    return coco_keypoint_results_file(results)


@pytest.fixture
def coco_keypoint_results_file_different_keypoint_lengths(
    coco_keypoint_results_file,
):
    """Return COCO results with different keypoint list lengths."""
    results = [
        COCO_KEYPOINT_RESULT_1,
        {
            **COCO_KEYPOINT_RESULT_2,
            "keypoints": [50, 60, 2],
        },
    ]
    return coco_keypoint_results_file(results)


@pytest.fixture
def coco_keypoint_annotations_file_different_keypoint_count(
    coco_keypoint_annotations_file,
):
    """Return COCO annotations with a different keypoint count."""
    categories = [
        {
            "id": 1,
            "name": "person",
            "keypoints": ["nose"],
        },
        {
            "id": 2,
            "name": "cat",
            "keypoints": ["nose"],
        },
    ]
    return coco_keypoint_annotations_file(categories)
