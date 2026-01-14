#!/usr/bin/env python3
"""Manual test script for poses_to_bboxes function."""

import numpy as np
import xarray as xr
import sys

# Add movement to path
sys.path.insert(0, '/home/eduardo/Projetos/Blockchain/movement')

from movement.transforms import poses_to_bboxes


def create_simple_poses_dataset():
    """Create a simple poses dataset with 2 individuals, 3 keypoints, 5 frames."""
    n_frames, n_space, n_keypoints, n_individuals = 5, 2, 3, 2

    # Create position data
    # Individual 0: keypoints at (0,0), (10,0), (10,10)
    # Individual 1: keypoints at (20,20), (30,20), (30,30)
    position = np.zeros((n_frames, n_space, n_keypoints, n_individuals))

    # Individual 0 keypoints (same for all frames for simplicity)
    for t in range(n_frames):
        position[t, 0, 0, 0] = 0.0  # x of keypoint 0
        position[t, 1, 0, 0] = 0.0  # y of keypoint 0
        position[t, 0, 1, 0] = 10.0  # x of keypoint 1
        position[t, 1, 1, 0] = 0.0  # y of keypoint 1
        position[t, 0, 2, 0] = 10.0  # x of keypoint 2
        position[t, 1, 2, 0] = 10.0  # y of keypoint 2

        # Individual 1 keypoints
        position[t, 0, 0, 1] = 20.0  # x of keypoint 0
        position[t, 1, 0, 1] = 20.0  # y of keypoint 0
        position[t, 0, 1, 1] = 30.0  # x of keypoint 1
        position[t, 1, 1, 1] = 20.0  # y of keypoint 1
        position[t, 0, 2, 1] = 30.0  # x of keypoint 2
        position[t, 1, 2, 1] = 30.0  # y of keypoint 2

    # Create confidence data (all 0.9)
    confidence = np.full((n_frames, n_keypoints, n_individuals), 0.9)

    # Create dataset
    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                position,
                dims=("time", "space", "keypoints", "individuals"),
            ),
            "confidence": xr.DataArray(
                confidence,
                dims=("time", "keypoints", "individuals"),
            ),
        },
        coords={
            "time": np.arange(n_frames),
            "space": ["x", "y"],
            "keypoints": ["kpt_0", "kpt_1", "kpt_2"],
            "individuals": ["id_0", "id_1"],
        },
        attrs={"fps": 30, "time_unit": "frames", "ds_type": "poses"},
    )

    return ds


def test_basic():
    """Test basic conversion from poses to bboxes."""
    print("=" * 60)
    print("Test 1: Basic conversion")
    print("=" * 60)

    ds = create_simple_poses_dataset()
    print(f"\nInput poses dataset:")
    print(f"  Dimensions: {ds.dims}")
    print(f"  Data variables: {list(ds.data_vars)}")

    result = poses_to_bboxes(ds)

    print(f"\nOutput bboxes dataset:")
    print(f"  Dimensions: {result.dims}")
    print(f"  Data variables: {list(result.data_vars)}")
    print(f"  ds_type: {result.attrs.get('ds_type')}")

    # Verify bbox calculations for individual 0
    # Keypoints: (0,0), (10,0), (10,10) -> bbox centroid (5, 5), shape (10, 10)
    print(f"\nIndividual 0 (frame 0):")
    print(f"  Centroid: ({result.position.values[0, 0, 0]:.1f}, {result.position.values[0, 1, 0]:.1f}) - Expected: (5.0, 5.0)")
    print(f"  Shape: ({result.shape.values[0, 0, 0]:.1f}, {result.shape.values[0, 1, 0]:.1f}) - Expected: (10.0, 10.0)")
    print(f"  Confidence: {result.confidence.values[0, 0]:.1f} - Expected: 0.9")

    # Verify bbox calculations for individual 1
    print(f"\nIndividual 1 (frame 0):")
    print(f"  Centroid: ({result.position.values[0, 0, 1]:.1f}, {result.position.values[0, 1, 1]:.1f}) - Expected: (25.0, 25.0)")
    print(f"  Shape: ({result.shape.values[0, 0, 1]:.1f}, {result.shape.values[0, 1, 1]:.1f}) - Expected: (10.0, 10.0)")
    print(f"  Confidence: {result.confidence.values[0, 1]:.1f} - Expected: 0.9")

    # Check assertions
    assert np.allclose(result.position.values[0, 0, 0], 5.0), "Individual 0 x centroid failed"
    assert np.allclose(result.position.values[0, 1, 0], 5.0), "Individual 0 y centroid failed"
    assert np.allclose(result.shape.values[0, 0, 0], 10.0), "Individual 0 width failed"
    assert np.allclose(result.shape.values[0, 1, 0], 10.0), "Individual 0 height failed"
    assert np.allclose(result.position.values[0, 0, 1], 25.0), "Individual 1 x centroid failed"
    assert np.allclose(result.position.values[0, 1, 1], 25.0), "Individual 1 y centroid failed"

    print("\n✓ Test 1 PASSED!\n")


def test_with_padding():
    """Test with padding."""
    print("=" * 60)
    print("Test 2: With padding")
    print("=" * 60)

    ds = create_simple_poses_dataset()
    padding_px = 5.0
    result = poses_to_bboxes(ds, padding_px=padding_px)

    expected_width = 10.0 + 2 * padding_px
    expected_height = 10.0 + 2 * padding_px

    print(f"\nPadding: {padding_px} pixels")
    print(f"Individual 0 (frame 0):")
    print(f"  Shape: ({result.shape.values[0, 0, 0]:.1f}, {result.shape.values[0, 1, 0]:.1f}) - Expected: ({expected_width:.1f}, {expected_height:.1f})")

    assert np.allclose(result.shape.values[0, 0, 0], expected_width), "Width with padding failed"
    assert np.allclose(result.shape.values[0, 1, 0], expected_height), "Height with padding failed"

    print("\n✓ Test 2 PASSED!\n")


def test_with_nan():
    """Test with NaN values."""
    print("=" * 60)
    print("Test 3: With NaN values")
    print("=" * 60)

    ds = create_simple_poses_dataset()

    # Set some keypoints to NaN
    ds.position.values[0, :, 0, 0] = np.nan  # First keypoint of ind 0, frame 0

    result = poses_to_bboxes(ds)

    # Frame 0, individual 0: only 2 keypoints valid
    # Remaining keypoints: (10,0), (10,10) -> centroid (10, 5), shape (0, 10)
    print(f"\nIndividual 0 (frame 0) with first keypoint as NaN:")
    print(f"  Centroid: ({result.position.values[0, 0, 0]:.1f}, {result.position.values[0, 1, 0]:.1f}) - Expected: (10.0, 5.0)")
    print(f"  Shape: ({result.shape.values[0, 0, 0]:.1f}, {result.shape.values[0, 1, 0]:.1f}) - Expected: (0.0, 10.0)")

    assert np.allclose(result.position.values[0, 0, 0], 10.0), "x centroid with NaN failed"
    assert np.allclose(result.position.values[0, 1, 0], 5.0), "y centroid with NaN failed"
    assert np.allclose(result.shape.values[0, 0, 0], 0.0), "width with NaN failed"
    assert np.allclose(result.shape.values[0, 1, 0], 10.0), "height with NaN failed"

    print("\n✓ Test 3 PASSED!\n")


def test_error_negative_padding():
    """Test error with negative padding."""
    print("=" * 60)
    print("Test 4: Error handling - negative padding")
    print("=" * 60)

    ds = create_simple_poses_dataset()

    try:
        result = poses_to_bboxes(ds, padding_px=-5)
        print("\n✗ Test 4 FAILED - Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"\n✓ Correctly raised ValueError: {e}")
        print("✓ Test 4 PASSED!\n")
        return True


def test_error_3d_poses():
    """Test error with 3D poses."""
    print("=" * 60)
    print("Test 5: Error handling - 3D poses")
    print("=" * 60)

    n_frames, n_space, n_keypoints, n_individuals = 2, 3, 2, 1
    position = np.full((n_frames, n_space, n_keypoints, n_individuals), 5.0)

    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                position,
                dims=("time", "space", "keypoints", "individuals"),
            ),
        },
        coords={
            "time": np.arange(n_frames),
            "space": ["x", "y", "z"],
            "keypoints": ["kpt_0", "kpt_1"],
            "individuals": ["id_0"],
        },
        attrs={"ds_type": "poses"},
    )

    try:
        result = poses_to_bboxes(ds)
        print("\n✗ Test 5 FAILED - Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"\n✓ Correctly raised ValueError: {e}")
        print("✓ Test 5 PASSED!\n")
        return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MANUAL TESTS FOR poses_to_bboxes()")
    print("=" * 60 + "\n")

    try:
        test_basic()
        test_with_padding()
        test_with_nan()
        test_error_negative_padding()
        test_error_3d_poses()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
