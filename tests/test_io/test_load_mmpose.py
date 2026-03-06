import json
import pytest
import numpy as np
import xarray as xr
from movement.io.load_poses import from_mmpose_file

def create_mmpose_json(filepath, frames_data):
    """Helper to create an MMPose JSON file."""
    with open(filepath, "w") as f:
        json.dump(frames_data, f)

def get_sample_instance(keypoints=None, scores=None, track_id=None):
    """Helper to create a sample instance dictionary."""
    return {
        "keypoints": keypoints or [[10.0, 20.0], [30.0, 40.0]],
        "keypoint_scores": scores or [0.9, 0.8],
        "bbox": [5, 5, 50, 50],
        "bbox_score": 0.95,
        "track_id": track_id
    }

def test_load_mmpose_single_frame(tmp_path):
    file_path = tmp_path / "mmpose_single.json"
    data = {"frame_id": 0, "instances": [get_sample_instance()]}
    create_mmpose_json(file_path, data)
    
    ds = from_mmpose_file(file_path)
    assert isinstance(ds, xr.Dataset)
    assert ds.position.shape == (1, 2, 2, 1)
    assert np.allclose(ds.position.values[0, :, 0, 0], [10.0, 20.0])

def test_load_mmpose_multi_frame(tmp_path):
    file_path = tmp_path / "mmpose_multi.json"
    data = [
        {"frame_id": 1, "instances": [get_sample_instance(keypoints=[[11, 21], [31, 41]], track_id=1)]},
        {"frame_id": 0, "instances": [get_sample_instance(track_id=1)]}
    ]
    create_mmpose_json(file_path, data)
    
    ds = from_mmpose_file(file_path)
    assert ds.position.shape == (2, 2, 2, 1)
    assert np.allclose(ds.position.values[0, :, 0, 0], [10.0, 20.0])
    assert np.allclose(ds.position.values[1, :, 0, 0], [11.0, 21.0])

def test_load_mmpose_multi_individual(tmp_path):
    file_path = tmp_path / "mmpose_multi_ind.json"
    data = {
        "frame_id": 0,
        "instances": [
            get_sample_instance(track_id=1),
            get_sample_instance(keypoints=[[100, 200], [300, 400]], track_id=2)
        ]
    }
    create_mmpose_json(file_path, data)
    
    ds = from_mmpose_file(file_path)
    assert ds.position.shape == (1, 2, 2, 2)
    assert "track_1" in ds.individuals.values
    assert "track_2" in ds.individuals.values

@pytest.mark.parametrize("invalid_content, match", [
    ("invalid json", "is not valid JSON"),
    ({"frame_id": 0}, "does not match schema")
])
def test_load_mmpose_errors(tmp_path, invalid_content, match):
    file_path = tmp_path / "error.json"
    with open(file_path, "w") as f:
        if isinstance(invalid_content, str):
            f.write(invalid_content)
        else:
            json.dump(invalid_content, f)
    
    with pytest.raises(ValueError, match=match):
        from_mmpose_file(file_path)
