from movement.io.export_via_tracks import export_via_tracks
import json
import os

def test_export_via_tracks():
    bboxes = {
        1: {0: {"x": 100, "y": 50, "width": 40, "height": 60}},
        2: {0: {"x": 110, "y": 55, "width": 42, "height": 62}}
    }
    export_via_tracks(bboxes, "test_via_tracks.json")
    
    with open("test_via_tracks.json", "r") as f:
        data = json.load(f)
    
    assert len(data) == 2
    assert "1" in data and "2" in data
    os.remove("test_via_tracks.json")
