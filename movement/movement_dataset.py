"""Accessor for extending :class:`xarray.Dataset` objects."""

import logging
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass
class MovementDataset:
    """A dataclass to define the canonical structure of a Movement Dataset."""

    # Set class attributes for expected dimensions and data variables
    dim_names: ClassVar[dict] = {
        "poses": ("time", "individuals", "keypoints", "space"),
        "bboxes": ("time", "individuals", "space"),
    }
    var_names: ClassVar[dict] = {
        "poses": ("position", "confidence"),
        "bboxes": ("position", "shape", "confidence"),
    }
