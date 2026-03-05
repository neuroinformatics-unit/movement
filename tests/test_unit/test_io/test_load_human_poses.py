import json

import numpy as np
import xarray as xr

from movement.io import load_poses


def test_from_mmpose_file(tmp_path):
    # Create a dummy MMPose JSON file
    mmpose_data = [
        {
            "instances": [
                {
                    "keypoints": [[10, 20], [30, 40]],
                    "keypoint_scores": [0.9, 0.8],
                },
                {
                    "keypoints": [[50, 60], [70, 80]],
                    "keypoint_scores": [0.7, 0.6],
                },
            ]
        },
        {
            "instances": [
                {
                    "keypoints": [[11, 21], [31, 41]],
                    "keypoint_scores": [0.95, 0.85],
                },
                {
                    "keypoints": [[51, 61], [71, 81]],
                    "keypoint_scores": [0.75, 0.65],
                },
            ]
        },
    ]
    file_path = tmp_path / "test_mmpose.json"
    with open(file_path, "w") as f:
        json.dump(mmpose_data, f)

    ds = load_poses.from_mmpose_file(file_path)
    assert isinstance(ds, xr.Dataset)
    assert ds.position.shape == (
        2,
        2,
        2,
        2,
    )  # (time, space, keypoints, individuals)
    assert ds.confidence.shape == (2, 2, 2)  # (time, keypoints, individuals)
    assert np.allclose(
        ds.position.sel(time=0, space="x", individuals="id_0"), [10, 30]
    )
    assert np.allclose(
        ds.confidence.sel(time=1, individuals="id_1"), [0.75, 0.65]
    )


def test_from_coco_file(tmp_path):
    # Create a dummy COCO JSON file
    coco_data = {
        "images": [{"id": 1}, {"id": 2}],
        "categories": [{"keypoints": ["nose", "eye"]}],
        "annotations": [
            {
                "image_id": 1,
                "id": 101,
                "track_id": 1,
                "keypoints": [10, 20, 0.9, 30, 40, 0.8],
            },
            {
                "image_id": 2,
                "id": 102,
                "track_id": 1,
                "keypoints": [11, 21, 0.95, 31, 41, 0.85],
            },
        ],
    }
    file_path = tmp_path / "test_coco.json"
    with open(file_path, "w") as f:
        json.dump(coco_data, f)

    ds = load_poses.from_coco_file(file_path)
    assert isinstance(ds, xr.Dataset)
    assert "nose" in ds.keypoints
    assert ds.position.shape == (2, 2, 2, 1)
    assert np.allclose(ds.position.sel(time=0, space="y", keypoints="eye"), 40)


def test_from_freemocap_dir(tmp_path):
    # Create a dummy FreeMoCap directory structure
    output_dir = tmp_path / "output_data"
    output_dir.mkdir()

    # Create a dummy CSV file
    # Columns: kp1_x, kp1_y, kp1_z, kp2_x, kp2_y, kp2_z
    csv_content = (
        "kp1_x,kp1_y,kp1_z,kp2_x,kp2_y,kp2_z\n"
        "1,2,3,4,5,6\n"
        "1.1,2.1,3.1,4.1,5.1,6.1"
    )
    csv_file = output_dir / "mediapipe_body_3d_xyz.csv"
    with open(csv_file, "w") as f:
        f.write(csv_content)

    ds = load_poses.from_freemocap_dir(tmp_path)
    assert isinstance(ds, xr.Dataset)
    assert ds.position.shape == (
        2,
        3,
        2,
        1,
    )  # (time, space, keypoints, individuals)
    assert "kp1" in ds.keypoints
    assert np.allclose(
        ds.position.sel(time=1, space="z", keypoints="kp2"), 6.1
    )


def test_from_motion_bids(tmp_path):
    # Create a dummy Motion-BIDS tsv file
    tsv_content = (
        "L_Ankle_x\tL_Ankle_y\tR_Ankle_x\tR_Ankle_y\n"
        "10\t20\t30\t40\n11\t21\t31\t41\n"
    )
    tsv_path = tmp_path / "test_motion.tsv"
    tsv_path.write_text(tsv_content)

    # sidecar json
    json_path = tmp_path / "test_motion.json"
    with open(json_path, "w") as f:
        json.dump({"SamplingFrequency": 30}, f)

    ds = load_poses.from_motion_bids(tsv_path)
    assert isinstance(ds, xr.Dataset)
    assert ds.position.shape == (2, 2, 2, 1)
    assert ds.fps == 30
    assert "L_Ankle" in ds.keypoints
    assert np.allclose(
        ds.position.sel(time=0, space="x", keypoints="L_Ankle"), 10
    )
