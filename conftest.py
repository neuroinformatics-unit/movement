"""Configuration for running the docstring examples as doctests."""

import numpy as np
import pytest
from _pytest.doctest import DoctestItem

from movement.io import load_poses


@pytest.fixture(autouse=True)
def add_doctest_namespace(request, doctest_namespace):
    """Provide a small poses dataset ``ds`` to the docstring examples.

    Many docstring examples refer to a poses dataset ``ds``, without
    showing how it was loaded. Here we create a synthetic one, with two
    individuals, three keypoints and a random-walk trajectory.
    A new dataset is created for each docstring, so that changes made
    in one example do not affect the others.
    """
    if not isinstance(request.node, DoctestItem):
        return
    rng = np.random.default_rng(seed=42)
    steps = rng.normal(scale=5, size=(200, 2, 3, 2))
    doctest_namespace["ds"] = load_poses.from_numpy(
        position_array=steps.cumsum(axis=0),
        individual_names=["ind1", "ind2"],
        keypoint_names=["key1", "key2", "key3"],
    )
