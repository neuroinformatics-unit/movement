"""Test suite for COCO and BVH loaders in the load_poses module."""

import json

import numpy as np
import pytest
import xarray as xr

from movement.io import load_poses
from movement.validators.datasets import ValidPosesInputs
from movement.validators.files import ValidBVHFile, ValidCOCOJSON

expected_values_poses = {
    "vars_dims": {"position": 4, "confidence": 3},
    "dim_names": ValidPosesInputs.DIM_NAMES,
}


# ============== COCO test fixtures ==================================


def _make_coco_data(
    n_images=3,
    n_individuals=2,
    n_keypoints=3,
    with_track_id=False,
    with_score=True,
):
    """Build a minimal COCO keypoint annotation dict."""
    keypoint_names = [
        "nose",
        "left_eye",
        "right_eye",
    ][:n_keypoints]
    images = [
        {"id": i, "file_name": f"frame_{i:04d}.jpg"} for i in range(n_images)
    ]
    annotations = []
    ann_id = 0
    for img in images:
        for ind in range(n_individuals):
            kps = []
            for k in range(n_keypoints):
                x = float(100 + ind * 50 + k * 10)
                y = float(200 + ind * 30 + k * 5)
                v = 2  # labelled and visible
                kps.extend([x, y, v])
            ann = {
                "id": ann_id,
                "image_id": img["id"],
                "category_id": 1,
                "keypoints": kps,
            }
            if with_track_id:
                ann["track_id"] = ind
            if with_score:
                ann["score"] = 0.9
            ann_id += 1
            annotations.append(ann)
    categories = [
        {
            "id": 1,
            "name": "person",
            "keypoints": keypoint_names,
        }
    ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


@pytest.fixture
def coco_json_file(tmp_path):
    """Return the path to a valid COCO keypoint JSON file."""
    data = _make_coco_data()
    file_path = tmp_path / "coco_keypoints.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def coco_json_file_with_track_id(tmp_path):
    """Return the path to a COCO JSON file with track IDs."""
    data = _make_coco_data(with_track_id=True)
    file_path = tmp_path / "coco_tracked.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def coco_json_file_invisible_keypoints(tmp_path):
    """Return path to COCO JSON with visibility=0 keypoints."""
    data = _make_coco_data(
        n_images=2,
        n_individuals=1,
        n_keypoints=3,
    )
    # Set first keypoint of first annotation to invisible
    data["annotations"][0]["keypoints"][2] = 0
    file_path = tmp_path / "coco_invisible.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def coco_json_file_single_individual(tmp_path):
    """Return path to a COCO JSON file with one individual."""
    data = _make_coco_data(n_individuals=1)
    file_path = tmp_path / "coco_single.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def coco_json_file_invalid_keypoints_length(tmp_path):
    """Return path to COCO JSON with wrong keypoints length."""
    data = _make_coco_data(n_images=1, n_individuals=1)
    # Corrupt keypoints: remove last value
    data["annotations"][0]["keypoints"] = [1.0, 2.0]
    file_path = tmp_path / "coco_bad_kps.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def coco_json_file_missing_keys(tmp_path):
    """Return path to a JSON file missing required COCO keys."""
    data = {"images": [], "annotations": []}
    file_path = tmp_path / "coco_missing_keys.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def coco_json_file_no_score(tmp_path):
    """Return path to a COCO JSON file without score field."""
    data = _make_coco_data(with_score=False)
    file_path = tmp_path / "coco_no_score.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


# ============== BVH test fixtures ==================================

_BVH_FRAME_0 = (
    "0.00 0.00 0.00 0.00 0.00 0.00 "
    "0.00 0.00 0.00 0.00 0.00 0.00 "
    "0.00 0.00 0.00 0.00 0.00 0.00"
)
_BVH_FRAME_1 = (
    "1.00 2.00 0.50 10.00 5.00 0.00 "
    "5.00 2.00 0.00 3.00 1.00 0.00 "
    "-3.00 1.00 0.00 3.00 -1.00 0.00"
)
_BVH_FRAME_2 = (
    "2.00 4.00 1.00 20.00 10.00 0.00 "
    "10.00 4.00 0.00 6.00 2.00 0.00 "
    "-6.00 2.00 0.00 6.00 -2.00 0.00"
)

SAMPLE_BVH = (
    "HIERARCHY\n"
    "ROOT Hips\n"
    "{\n"
    "  OFFSET 0.00 0.00 0.00\n"
    "  CHANNELS 6 Xposition Yposition Zposition"
    " Zrotation Xrotation Yrotation\n"
    "  JOINT Spine\n"
    "  {\n"
    "    OFFSET 0.00 5.21 0.00\n"
    "    CHANNELS 3 Zrotation Xrotation Yrotation\n"
    "    JOINT Head\n"
    "    {\n"
    "      OFFSET 0.00 5.45 0.00\n"
    "      CHANNELS 3 Zrotation Xrotation"
    " Yrotation\n"
    "      End Site\n"
    "      {\n"
    "        OFFSET 0.00 3.00 0.00\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  JOINT LeftArm\n"
    "  {\n"
    "    OFFSET 3.50 4.80 0.00\n"
    "    CHANNELS 3 Zrotation Xrotation Yrotation\n"
    "    End Site\n"
    "    {\n"
    "      OFFSET 5.00 0.00 0.00\n"
    "    }\n"
    "  }\n"
    "  JOINT RightArm\n"
    "  {\n"
    "    OFFSET -3.50 4.80 0.00\n"
    "    CHANNELS 3 Zrotation Xrotation Yrotation\n"
    "    End Site\n"
    "    {\n"
    "      OFFSET -5.00 0.00 0.00\n"
    "    }\n"
    "  }\n"
    "}\n"
    "MOTION\n"
    "Frames: 3\n"
    "Frame Time: 0.033333\n"
    f"{_BVH_FRAME_0}\n"
    f"{_BVH_FRAME_1}\n"
    f"{_BVH_FRAME_2}\n"
)


@pytest.fixture
def bvh_file(tmp_path):
    """Return the path to a valid BVH file."""
    file_path = tmp_path / "motion.bvh"
    with open(file_path, "w") as f:
        f.write(SAMPLE_BVH)
    return file_path


@pytest.fixture
def bvh_file_no_hierarchy(tmp_path):
    """Return path to a BVH file missing HIERARCHY."""
    file_path = tmp_path / "bad_no_hierarchy.bvh"
    with open(file_path, "w") as f:
        f.write("MOTION\nFrames: 1\nFrame Time: 0.03\n0 0 0\n")
    return file_path


@pytest.fixture
def bvh_file_no_motion(tmp_path):
    """Return path to a BVH file missing MOTION section."""
    file_path = tmp_path / "bad_no_motion.bvh"
    with open(file_path, "w") as f:
        f.write("HIERARCHY\nROOT Hips\n{\nOFFSET 0 0 0\n}\n")
    return file_path


# ============== COCO loader tests ==================================


class TestCOCOLoader:
    """Tests for the COCO keypoint loader."""

    def test_load_from_coco_file(self, coco_json_file, helpers):
        """Test loading COCO keypoints returns valid Dataset."""
        ds = load_poses.from_coco_file(coco_json_file, fps=30)
        expected_values = {
            **expected_values_poses,
            "source_software": "COCO",
            "file_path": coco_json_file,
            "fps": 30,
        }
        helpers.assert_valid_dataset(ds, expected_values)

    def test_coco_dataset_shape(self, coco_json_file):
        """Test that COCO dataset has expected shape."""
        ds = load_poses.from_coco_file(coco_json_file)
        # 3 images, 2 space dims, 3 keypoints, 2 individuals
        assert ds.position.shape == (3, 2, 3, 2)
        assert ds.confidence.shape == (3, 3, 2)

    def test_coco_keypoint_names(self, coco_json_file):
        """Test that keypoint names match categories."""
        ds = load_poses.from_coco_file(coco_json_file)
        assert ds.coords["keypoints"].values.tolist() == [
            "nose",
            "left_eye",
            "right_eye",
        ]

    def test_coco_with_track_id(self, coco_json_file_with_track_id):
        """Test COCO loading with track_id for individuals."""
        ds = load_poses.from_coco_file(coco_json_file_with_track_id)
        assert ds.coords["individuals"].values.tolist() == [
            "id_0",
            "id_1",
        ]
        # All positions should be finite (all visible)
        assert not np.isnan(ds.position.values).all()

    def test_coco_invisible_keypoints(
        self, coco_json_file_invisible_keypoints
    ):
        """Test that invisible keypoints get NaN position."""
        ds = load_poses.from_coco_file(coco_json_file_invisible_keypoints)
        # First keypoint of first frame/individual is invisible
        pos = ds.position.sel(
            time=0,
            keypoints="nose",
            individuals="id_0",
        )
        assert np.isnan(pos.values).all()
        # Confidence for invisible keypoint should be 0
        conf = ds.confidence.sel(
            time=0,
            keypoints="nose",
            individuals="id_0",
        )
        assert conf.values == pytest.approx(0.0)

    def test_coco_single_individual(self, coco_json_file_single_individual):
        """Test loading COCO file with one individual."""
        ds = load_poses.from_coco_file(coco_json_file_single_individual)
        assert ds.sizes["individuals"] == 1

    def test_coco_no_score_field(self, coco_json_file_no_score):
        """Test loading COCO without score field."""
        ds = load_poses.from_coco_file(coco_json_file_no_score)
        # With v=2 and no score, confidence = 2/2 * 1.0 = 1.0
        conf = ds.confidence.values
        visible_mask = ~np.isnan(conf)
        assert np.allclose(conf[visible_mask], 1.0)

    def test_coco_fps_none(self, coco_json_file):
        """Test that fps=None gives frame-number time coords."""
        ds = load_poses.from_coco_file(coco_json_file, fps=None)
        assert ds.time_unit == "frames"

    def test_coco_fps_set(self, coco_json_file):
        """Test that providing fps gives second-based coords."""
        ds = load_poses.from_coco_file(coco_json_file, fps=30)
        assert ds.time_unit == "seconds"
        assert ds.fps == 30

    def test_coco_source_software_attr(self, coco_json_file):
        """Test source_software attribute is set correctly."""
        ds = load_poses.from_coco_file(coco_json_file)
        assert ds.source_software == "COCO"

    def test_coco_source_file_attr(self, coco_json_file):
        """Test source_file attribute is set correctly."""
        ds = load_poses.from_coco_file(coco_json_file)
        assert ds.source_file == coco_json_file.as_posix()


class TestCOCOValidation:
    """Tests for COCO file validation."""

    def test_valid_coco_json(self, coco_json_file):
        """Test that valid COCO JSON passes validation."""
        valid = ValidCOCOJSON(file=coco_json_file)
        assert valid.file == coco_json_file

    def test_invalid_coco_missing_keys(self, coco_json_file_missing_keys):
        """Test that JSON missing COCO keys fails."""
        with pytest.raises(ValueError, match="schema"):
            ValidCOCOJSON(file=coco_json_file_missing_keys)

    def test_invalid_coco_keypoints_length(
        self, coco_json_file_invalid_keypoints_length
    ):
        """Test that wrong keypoints array length fails."""
        with pytest.raises(ValueError, match="keypoint"):
            ValidCOCOJSON(
                file=coco_json_file_invalid_keypoints_length,
            )

    def test_invalid_coco_wrong_extension(self, wrong_extension_file):
        """Test that wrong file extension fails."""
        with pytest.raises(ValueError, match="suffix"):
            ValidCOCOJSON(file=wrong_extension_file)


# ============== BVH loader tests ===================================


class TestBVHLoader:
    """Tests for the BVH file loader."""

    def test_load_from_bvh_file(self, bvh_file, helpers):
        """Test loading BVH file returns valid Dataset."""
        ds = load_poses.from_bvh_file(bvh_file)
        expected_values = {
            **expected_values_poses,
            "source_software": "BVH",
            "file_path": bvh_file,
        }
        helpers.assert_valid_dataset(ds, expected_values)

    def test_bvh_dataset_shape(self, bvh_file):
        """Test that BVH dataset has expected shape."""
        ds = load_poses.from_bvh_file(bvh_file)
        # 3 frames, 3 space dims, 5 joints, 1 individual
        assert ds.position.shape == (3, 3, 5, 1)
        assert ds.confidence.shape == (3, 5, 1)

    def test_bvh_joint_names(self, bvh_file):
        """Test that joint names match BVH hierarchy."""
        ds = load_poses.from_bvh_file(bvh_file)
        expected_joints = [
            "Hips",
            "Spine",
            "Head",
            "LeftArm",
            "RightArm",
        ]
        actual = ds.coords["keypoints"].values.tolist()
        assert actual == expected_joints

    def test_bvh_3d_space(self, bvh_file):
        """Test that BVH data has 3 spatial dimensions."""
        ds = load_poses.from_bvh_file(bvh_file)
        assert ds.sizes["space"] == 3
        assert "z" in ds.coords["space"].values

    def test_bvh_fps_from_frame_time(self, bvh_file):
        """Test fps is computed from BVH Frame Time."""
        ds = load_poses.from_bvh_file(bvh_file)
        # Frame Time: 0.033333 → fps ≈ 30
        assert ds.fps == pytest.approx(30.0, abs=0.1)
        assert ds.time_unit == "seconds"

    def test_bvh_fps_override(self, bvh_file):
        """Test that providing fps overrides Frame Time."""
        ds = load_poses.from_bvh_file(bvh_file, fps=60)
        assert ds.fps == 60
        assert ds.time_unit == "seconds"

    def test_bvh_root_position_frame_0(self, bvh_file):
        """Test root position in the rest pose (frame 0)."""
        ds = load_poses.from_bvh_file(bvh_file)
        root_pos = ds.position.sel(
            time=ds.coords["time"][0],
            keypoints="Hips",
            individuals="id_0",
        )
        # Frame 0: all zeros in channels, offset 0,0,0
        np.testing.assert_allclose(root_pos.values, [0.0, 0.0, 0.0], atol=1e-6)

    def test_bvh_root_position_frame_1(self, bvh_file):
        """Test root position in frame 1 (translation)."""
        ds = load_poses.from_bvh_file(bvh_file)
        root_pos = ds.position.sel(
            time=ds.coords["time"][1],
            keypoints="Hips",
            individuals="id_0",
        )
        # Frame 1: Xposition=1, Yposition=2, Zposition=0.5
        np.testing.assert_allclose(root_pos.values, [1.0, 2.0, 0.5], atol=1e-6)

    def test_bvh_source_software_attr(self, bvh_file):
        """Test source_software attribute is set correctly."""
        ds = load_poses.from_bvh_file(bvh_file)
        assert ds.source_software == "BVH"

    def test_bvh_source_file_attr(self, bvh_file):
        """Test source_file attribute is set correctly."""
        ds = load_poses.from_bvh_file(bvh_file)
        assert ds.source_file == bvh_file.as_posix()

    def test_bvh_confidence_is_nan(self, bvh_file):
        """Test BVH has NaN confidence (no conf info)."""
        ds = load_poses.from_bvh_file(bvh_file)
        assert np.isnan(ds.confidence.values).all()

    def test_bvh_single_individual(self, bvh_file):
        """Test BVH creates single-individual dataset."""
        ds = load_poses.from_bvh_file(bvh_file)
        assert ds.sizes["individuals"] == 1


class TestBVHValidation:
    """Tests for BVH file validation."""

    def test_valid_bvh_file(self, bvh_file):
        """Test that a valid BVH file passes validation."""
        valid = ValidBVHFile(file=bvh_file)
        assert valid.file == bvh_file

    def test_invalid_bvh_no_hierarchy(self, bvh_file_no_hierarchy):
        """Test BVH without HIERARCHY fails validation."""
        with pytest.raises(ValueError, match="HIERARCHY"):
            ValidBVHFile(file=bvh_file_no_hierarchy)

    def test_invalid_bvh_no_motion(self, bvh_file_no_motion):
        """Test BVH without MOTION fails validation."""
        with pytest.raises(ValueError, match="MOTION"):
            ValidBVHFile(file=bvh_file_no_motion)

    def test_invalid_bvh_wrong_extension(self, wrong_extension_file):
        """Test that wrong file extension fails."""
        with pytest.raises(ValueError, match="suffix"):
            ValidBVHFile(file=wrong_extension_file)


# ============== load_dataset integration tests =====================


class TestLoadDatasetIntegration:
    """Test that COCO and BVH work through load_dataset."""

    def test_load_dataset_coco(self, coco_json_file):
        """Test load_dataset with source_software='COCO'."""
        from movement.io import load_dataset

        ds = load_dataset(
            coco_json_file,
            source_software="COCO",
            fps=30,
        )
        assert isinstance(ds, xr.Dataset)
        assert ds.source_software == "COCO"

    def test_load_dataset_bvh(self, bvh_file):
        """Test load_dataset with source_software='BVH'."""
        from movement.io import load_dataset

        ds = load_dataset(bvh_file, source_software="BVH")
        assert isinstance(ds, xr.Dataset)
        assert ds.source_software == "BVH"
