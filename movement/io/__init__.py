from . import load_bboxes, load_poses  # Trigger register_loader decorators
from .load import load_dataset, load_multiview_dataset

__all__ = ["load_dataset", "load_multiview_dataset"]
