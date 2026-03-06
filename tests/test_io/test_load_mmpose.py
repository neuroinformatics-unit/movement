import json
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from movement.io.load_poses import from_mmpose_file

def test_load_mmpose_single_frame(tmp_path):
    data = {
        "frame_id": 0,
        "instances": [
            {
                "keypoints": [[10.0, 20.0], [30.0, 40.0]],
                "keypoint_scores": [0.9, 0.8],
                "bbox": [5, 5, 50, 50],
                "bbox_score": 0.95
            }
        ]
    }
    file_path = tmp_path / "mmpose_single.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    
    ds = from_mmpose_file(file_path)
    assert isinstance(ds, xr.Dataset)
    assert ds.attrs["source_software"] == "MMPose"
    assert ds.position.shape == (1, 2, 2, 1)  # (time, space, keypoints, individuals)
    assert np.allclose(ds.position.values[0, :, 0, 0], [10.0, 20.0])
    assert np.allclose(ds.confidence.values[0, 0, 0], 0.9)

def test_load_mmpose_multi_frame(tmp_path):
    data = [
        {
            "frame_id": 1,
            "instances": [
                {
                    "keypoints": [[11.0, 21.0], [31.0, 41.0]],
                    "keypoint_scores": [0.91, 0.81],
                    "track_id": 1
                }
            ]
        },
        {
            "frame_id": 0,
            "instances": [
                {
                    "keypoints": [[10.0, 20.0], [30.0, 40.0]],
                    "keypoint_scores": [0.9, 0.8],
                    "track_id": 1
                }
            ]
        }
    ]
    file_path = tmp_path / "mmpose_multi.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    
    ds = from_mmpose_file(file_path)
    assert ds.position.shape == (2, 2, 2, 1)
    # Check sorting by frame_id
    assert np.allclose(ds.position.values[0, :, 0, 0], [10.0, 20.0])
    assert np.allclose(ds.position.values[1, :, 0, 0], [11.0, 21.0])
    assert "track_1" in ds.individuals.values

def test_load_mmpose_multi_individual(tmp_path):
    data = {
        "frame_id": 0,
        "instances": [
            {
                "keypoints": [[10.0, 20.0], [30.0, 40.0]],
                "keypoint_scores": [0.9, 0.8],
                "track_id": 1
            },
            {
                "keypoints": [[100.0, 200.0], [300.0, 400.0]],
                "keypoint_scores": [0.7, 0.6],
                "track_id": 2
            }
        ]
    }
    file_path = tmp_path / "mmpose_multi_ind.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    
    ds = from_mmpose_file(file_path)
    assert ds.position.shape == (1, 2, 2, 2)
    assert "track_1" in ds.individuals.values
    assert "track_2" in ds.individuals.values
    assert np.allclose(ds.position.values[0, :, 0, 0], [10.0, 20.0])
    assert np.allclose(ds.position.values[0, :, 0, 1], [100.0, 200.0])

def test_load_mmpose_invalid_json(tmp_path):
    file_path = tmp_path / "invalid.json"
    with open(file_path, "w") as f:
        f.write("invalid json")
    
    with pytest.raises(ValueError, match="is not valid JSON"):
        from_mmpose_file(file_path)

def test_load_mmpose_schema_mismatch(tmp_path):
    data = {"frame_id": 0}  # Missing instances
    file_path = tmp_path / "mismatch.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    
    with pytest.raises(ValueError, match="does not match schema"):
        from_mmpose_file(file_path)
