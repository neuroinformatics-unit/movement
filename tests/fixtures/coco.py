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
