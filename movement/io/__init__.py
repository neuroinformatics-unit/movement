from . import load_aniframe, load_bboxes, load_poses  # Trigger register_loader decorators
from .load import infer_source_software, load_dataset, load_multiview_dataset

__all__ = ["infer_source_software", "load_dataset", "load_multiview_dataset"]
