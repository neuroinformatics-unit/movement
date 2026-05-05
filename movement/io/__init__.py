from . import load_bboxes, load_poses  # Trigger register_loader decorators
from .load import (
    get_supported_source_software,
    infer_source_software,
    load_dataset,
    load_multiview_dataset,
    rename_legacy_dimensions,
)

__all__ = [
    "get_supported_source_software",
    "infer_source_software",
    "load_dataset",
    "load_multiview_dataset",
    "rename_legacy_dimensions",
]
