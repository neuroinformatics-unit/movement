import logging

import pandas as pd
from napari.settings import get_settings

from movement.logging import log_error

logger = logging.getLogger(__name__)


def columns_to_categorical_codes(
    df: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """Convert columns in a DataFrame to ordered categorical codes. The codes
    are integers corresponding to the unique values in the column,
    ordered by appearance.
    """
    new_df = df.copy()
    for col in cols:
        cat_dtype = pd.api.types.CategoricalDtype(
            categories=df[col].unique().tolist(), ordered=True
        )
        new_df[col] = df[col].astype(cat_dtype).cat.codes
    return new_df


def set_playback_fps(fps: int):
    """Set the playback speed for the napari viewer."""
    # Check that the passed fps is a positive integer > 0 and < 1000
    if not isinstance(fps, int) or fps < 1 or fps > 1000:
        raise log_error(
            ValueError,
            "Playback fps must be a positive integer between 1 and 1000.",
        )
    settings = get_settings()
    settings.application.playback_fps = fps
