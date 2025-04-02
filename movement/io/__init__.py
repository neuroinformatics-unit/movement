"""Input/output functionality for the `movement` package."""

import warnings
from typing import Any

from movement.io.load_dataset import (
    from_anipose_file,
    from_anipose_style_df,
    from_dlc_file,
    from_dlc_style_df,
    from_file,
    from_lp_file,
    from_multiview_files,
    from_numpy,
    from_sleap_file,
)
from movement.io.save_dataset import (
    to_dlc_file,
    to_dlc_style_df,
    to_lp_file,
    to_sleap_analysis_file,
)

# For backward compatibility
from movement.io.load_poses import (
    from_anipose_file as _from_anipose_file,
    from_anipose_style_df as _from_anipose_style_df,
    from_dlc_file as _from_dlc_file,
    from_dlc_style_df as _from_dlc_style_df,
    from_file as _from_file,
    from_lp_file as _from_lp_file,
    from_multiview_files as _from_multiview_files,
    from_numpy as _from_numpy,
    from_sleap_file as _from_sleap_file,
)
from movement.io.save_poses import (
    to_dlc_file as _to_dlc_file,
    to_dlc_style_df as _to_dlc_style_df,
    to_lp_file as _to_lp_file,
    to_sleap_analysis_file as _to_sleap_analysis_file,
)


def _deprecation_warning(old_name: str, new_name: str) -> None:
    """Emit a deprecation warning for old function names."""
    warnings.warn(
        f"The function `{old_name}` is deprecated and will be removed in a future "
        f"version. Please use `{new_name}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


# Deprecated functions that forward to new ones
def from_anipose_file(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.load_dataset.from_anipose_file instead."""
    _deprecation_warning(
        "movement.io.load_poses.from_anipose_file",
        "movement.io.load_dataset.from_anipose_file",
    )
    return _from_anipose_file(*args, **kwargs)


def from_anipose_style_df(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.load_dataset.from_anipose_style_df instead."""
    _deprecation_warning(
        "movement.io.load_poses.from_anipose_style_df",
        "movement.io.load_dataset.from_anipose_style_df",
    )
    return _from_anipose_style_df(*args, **kwargs)


def from_dlc_file(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.load_dataset.from_dlc_file instead."""
    _deprecation_warning(
        "movement.io.load_poses.from_dlc_file",
        "movement.io.load_dataset.from_dlc_file",
    )
    return _from_dlc_file(*args, **kwargs)


def from_dlc_style_df(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.load_dataset.from_dlc_style_df instead."""
    _deprecation_warning(
        "movement.io.load_poses.from_dlc_style_df",
        "movement.io.load_dataset.from_dlc_style_df",
    )
    return _from_dlc_style_df(*args, **kwargs)


def from_file(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.load_dataset.from_file instead."""
    _deprecation_warning(
        "movement.io.load_poses.from_file",
        "movement.io.load_dataset.from_file",
    )
    return _from_file(*args, **kwargs)


def from_lp_file(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.load_dataset.from_lp_file instead."""
    _deprecation_warning(
        "movement.io.load_poses.from_lp_file",
        "movement.io.load_dataset.from_lp_file",
    )
    return _from_lp_file(*args, **kwargs)


def from_multiview_files(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.load_dataset.from_multiview_files instead."""
    _deprecation_warning(
        "movement.io.load_poses.from_multiview_files",
        "movement.io.load_dataset.from_multiview_files",
    )
    return _from_multiview_files(*args, **kwargs)


def from_numpy(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.load_dataset.from_numpy instead."""
    _deprecation_warning(
        "movement.io.load_poses.from_numpy",
        "movement.io.load_dataset.from_numpy",
    )
    return _from_numpy(*args, **kwargs)


def from_sleap_file(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.load_dataset.from_sleap_file instead."""
    _deprecation_warning(
        "movement.io.load_poses.from_sleap_file",
        "movement.io.load_dataset.from_sleap_file",
    )
    return _from_sleap_file(*args, **kwargs)


def to_dlc_file(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.save_dataset.to_dlc_file instead."""
    _deprecation_warning(
        "movement.io.save_poses.to_dlc_file",
        "movement.io.save_dataset.to_dlc_file",
    )
    return _to_dlc_file(*args, **kwargs)


def to_dlc_style_df(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.save_dataset.to_dlc_style_df instead."""
    _deprecation_warning(
        "movement.io.save_poses.to_dlc_style_df",
        "movement.io.save_dataset.to_dlc_style_df",
    )
    return _to_dlc_style_df(*args, **kwargs)


def to_lp_file(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.save_dataset.to_lp_file instead."""
    _deprecation_warning(
        "movement.io.save_poses.to_lp_file",
        "movement.io.save_dataset.to_lp_file",
    )
    return _to_lp_file(*args, **kwargs)


def to_sleap_analysis_file(*args: Any, **kwargs: Any) -> Any:
    """Deprecated: Use movement.io.save_dataset.to_sleap_analysis_file instead."""
    _deprecation_warning(
        "movement.io.save_poses.to_sleap_analysis_file",
        "movement.io.save_dataset.to_sleap_analysis_file",
    )
    return _to_sleap_analysis_file(*args, **kwargs)

__all__ = [
    # Load functions
    "from_anipose_file",
    "from_anipose_style_df",
    "from_dlc_file",
    "from_dlc_style_df",
    "from_file",
    "from_lp_file",
    "from_multiview_files",
    "from_numpy",
    "from_sleap_file",
    # Save functions
    "to_dlc_file",
    "to_dlc_style_df",
    "to_lp_file",
    "to_sleap_analysis_file",
]
