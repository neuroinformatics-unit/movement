import pandas as pd
import pytest
from napari.settings import get_settings

from movement.napari.utils import (
    columns_to_categorical_codes,
    set_playback_fps,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "numeric": [1, 4, 3, 1],
            "string": ["some", "alpha", "zulu", "some"],
            "mixed": ["some", 2, 3, "some"],
        }
    )


@pytest.mark.parametrize(
    "cols", [["numeric"], ["string"], ["numeric", "string", "mixed"]]
)
def test_columns_to_categorical_codes(df, cols):
    """Test that the passed columns are converted to categorical codes
    ordered by appearance.
    """
    new_df = columns_to_categorical_codes(df, cols)
    for col in df.columns:
        if col in cols:
            pd.testing.assert_series_equal(
                new_df[col],
                pd.Series([0, 1, 2, 0], name=col, dtype="int8"),
            )
        else:
            pd.testing.assert_series_equal(new_df[col], df[col])


@pytest.mark.parametrize("fps", [-1, 0, 10, "str", 10.1, 60, 2000])
def test_set_playback_fps(fps):
    """Test that the playback speed for the napari viewer is set."""
    if fps in [10, 60]:  # valid fps values
        set_playback_fps(fps)
        settings = get_settings()
        assert settings.application.playback_fps == fps
    else:  # invalid fps values
        with pytest.raises(
            ValueError, match="positive integer between 1 and 1000."
        ):
            set_playback_fps(fps)
