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
    the given categories, and optional images and annotations to a JSON file.
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
    """Return a valid COCO keypoint results file with two detections
    (categories 1 and 2) in image 10 and one detection (category 1)
    in image 20.
    """
    results = [
        COCO_KEYPOINT_RESULT_1,
        COCO_KEYPOINT_RESULT_2,
        {
            "image_id": 20,
            "category_id": 1,
            "keypoints": [15, 25, 2, 35, 45, 2],
            "score": 0.7,
        },
    ]
    return coco_keypoint_results_file(results)


@pytest.fixture
def coco_keypoint_results_file_categories_out_of_order(
    coco_keypoint_results_file,
):
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
def coco_keypoint_annotations_file_valid(
    coco_keypoint_annotations_file,
):
    """Return valid COCO annotations with three categories sharing
    the same keypoints.
    """
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
def coco_keypoint_results_file_single_detection(coco_keypoint_results_file):
    """Return COCO keypoint results with a single detection."""
    return coco_keypoint_results_file([COCO_KEYPOINT_RESULT_1])
