from types import SimpleNamespace

import pandas as pd
import pytest

from movement.utils.napari_utils import set_tracks_color_by


@pytest.fixture
def mock_tracks_layer():
    # Simulate a napari Tracks layer with a `.features` attribute
    return SimpleNamespace(
        features=pd.DataFrame(
            {"individual_factorized": [0, 1, 2], "track_id": [1, 2, 3]}
        ),
        color_by=None,
    )


def test_set_tracks_color_by_preferred_found(mock_tracks_layer):
    set_tracks_color_by(mock_tracks_layer, preferred="individual_factorized")
    assert mock_tracks_layer.color_by == "individual_factorized"


def test_set_tracks_color_by_preferred_missing(mock_tracks_layer):
    set_tracks_color_by(mock_tracks_layer, preferred="nonexistent_feature")
    assert mock_tracks_layer.color_by == "track_id"


def test_set_tracks_color_by_gui_overlay(mock_tracks_layer):
    mock_viewer = SimpleNamespace(text_overlay=SimpleNamespace(text=""))
    set_tracks_color_by(
        mock_tracks_layer, preferred="missing", viewer=mock_viewer
    )
    assert "Using 'track_id'" in mock_viewer.text_overlay.text
