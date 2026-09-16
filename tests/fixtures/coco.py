"""COCO keypoint results and annotations fixtures."""

import json

import pytest


@pytest.fixture
def coco_results_file(tmp_path):
    """Return a function to create a COCO keypoint results file."""

    def _coco_results_file(results):
        file_path = tmp_path / "coco_results.json"

        with open(file_path, "w") as f:
            json.dump(results, f)

        return file_path

    return _coco_results_file


@pytest.fixture
def coco_annotations_file(tmp_path):
    """Return a function to create a COCO annotations file."""

    def _coco_annotations_file(categories):
        file_path = tmp_path / "coco_annotations.json"

        with open(file_path, "w") as f:
            json.dump({"categories": categories}, f)

        return file_path

    return _coco_annotations_file


@pytest.fixture
def coco_keypoints_file(coco_results_file):
    """Return a valid COCO keypoint results file."""
    results = [
        {
            "image_id": 10,
            "category_id": 1,
            "keypoints": [10, 20, 2, 30, 40, 2],
            "score": 0.9,
        },
        {
            "image_id": 10,
            "category_id": 2,
            "keypoints": [50, 60, 2, 70, 80, 2],
            "score": 0.8,
        },
        {
            "image_id": 20,
            "category_id": 1,
            "keypoints": [15, 25, 2, 35, 45, 2],
            "score": 0.7,
        },
    ]
    return coco_results_file(results)


@pytest.fixture
def coco_keypoints_file_category_as_track(coco_results_file):
    """Return COCO keypoint results for category-as-track tests."""
    results = [
        {
            "image_id": 10,
            "category_id": 2,
            "keypoints": [50, 60, 2, 70, 80, 2],
            "score": 0.8,
        },
        {
            "image_id": 10,
            "category_id": 1,
            "keypoints": [10, 20, 2, 30, 40, 2],
            "score": 0.9,
        },
    ]
    return coco_results_file(results)


@pytest.fixture
def coco_annotations_file_category_as_track(coco_annotations_file):
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
    return coco_annotations_file(categories)


@pytest.fixture
def coco_keypoints_file_duplicate_category(coco_results_file):
    """Return COCO results with duplicate category in a frame."""
    results = [
        {
            "image_id": 10,
            "category_id": 1,
            "keypoints": [10, 20, 2, 30, 40, 2],
            "score": 0.9,
        },
        {
            "image_id": 10,
            "category_id": 1,
            "keypoints": [50, 60, 2, 70, 80, 2],
            "score": 0.8,
        },
    ]
    return coco_results_file(results)


@pytest.fixture
def coco_keypoints_file_unknown_category(coco_results_file):
    """Return COCO results with an unknown category."""
    results = [
        {
            "image_id": 10,
            "category_id": 3,
            "keypoints": [10, 20, 2, 30, 40, 2],
            "score": 0.9,
        },
    ]
    return coco_results_file(results)


@pytest.fixture
def coco_keypoints_file_person(coco_results_file):
    """Return COCO keypoint results containing a person."""
    results = [
        {
            "image_id": 10,
            "category_id": 1,
            "keypoints": [10, 20, 2, 30, 40, 2],
            "score": 0.9,
        },
    ]
    return coco_results_file(results)


@pytest.fixture
def coco_annotations_file_person(coco_annotations_file):
    """Return COCO annotations containing a person category."""
    categories = [
        {
            "id": 1,
            "name": "person",
            "keypoints": ["nose", "left_eye"],
        },
    ]
    return coco_annotations_file(categories)


@pytest.fixture
def coco_keypoints_file_different_skeleton(coco_results_file):
    """Return COCO results using two categories."""
    results = [
        {
            "image_id": 10,
            "category_id": 1,
            "keypoints": [10, 20, 2, 30, 40, 2],
            "score": 0.9,
        },
        {
            "image_id": 10,
            "category_id": 2,
            "keypoints": [50, 60, 2, 70, 80, 2],
            "score": 0.8,
        },
    ]
    return coco_results_file(results)


@pytest.fixture
def coco_annotations_file_different_skeleton(coco_annotations_file):
    """Return COCO annotations with different keypoint skeletons."""
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
    return coco_annotations_file(categories)


@pytest.fixture
def coco_keypoints_file_without_annotations(coco_results_file):
    """Return COCO keypoint results without annotations."""
    results = [
        {
            "image_id": 20,
            "category_id": 5,
            "keypoints": [10, 20, 2, 30, 40, 2],
            "score": 0.9,
        },
    ]
    return coco_results_file(results)
