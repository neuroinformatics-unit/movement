"""Unit tests for nan reporting utilities."""

from movement.utils.reports import calculate_nan_stats, report_nan_values

# Tests -------------------------------------------------------------------


class TestCalculateNanStats:
    """Test suite for calculate_nan_stats function."""

    def test_full_dims(self, valid_poses_dataset_with_nan):
        """Test calculation with all dimensions present."""
        result = calculate_nan_stats(
            valid_poses_dataset_with_nan.position, "centroid", "id_0"
        )
        assert "centroid:" in result
        assert "/10" in result  # Total frames is 10

    def test_no_space(self, valid_poses_dataset_with_nan):
        """Test calculation without space dimension."""
        # Reduce dimensions by taking mean across space
        reduced_data = valid_poses_dataset_with_nan.position.mean("space")
        result = calculate_nan_stats(reduced_data, "centroid", "id_0")
        assert "centroid:" in result
        assert "/10" in result

    def test_minimal(self, valid_poses_dataset_with_nan):
        """Test calculation with minimal data (time only)."""
        # Reduce to time dimension only by taking mean across other dims
        minimal_data = valid_poses_dataset_with_nan.position.mean(
            ["space", "keypoints", "individuals"]
        )
        result = calculate_nan_stats(minimal_data)
        assert "data:" in result
        assert "/10" in result

    def test_all_keypoints_case(self, valid_poses_dataset_with_nan):
        """Test calculation when no specific keypoint is given."""
        result = calculate_nan_stats(
            valid_poses_dataset_with_nan.position,
            keypoint=None,
            individual="id_0",
        )
        assert "all_keypoints:" in result
        assert "/10" in result


class TestReportNanValues:
    """Test suite for report_nan_values function."""

    def test_full_dims(self, valid_poses_dataset_with_nan):
        """Test report generation with all dimensions."""
        report = report_nan_values(valid_poses_dataset_with_nan.position)
        assert "Individual: id_0" in report
        assert "centroid:" in report
        assert "(any spatial coordinate)" in report

    def test_no_space(self, valid_poses_dataset_with_nan):
        """Test report without space dimension."""
        # Reduce dimensions by taking mean across space
        reduced_data = valid_poses_dataset_with_nan.position.mean("space")
        report = report_nan_values(reduced_data)
        assert "Individual: id_0" in report
        assert "centroid:" in report
        assert "(any spatial coordinate)" not in report

    def test_minimal(self, valid_poses_dataset_with_nan):
        """Test report with minimal data (time only)."""
        # Reduce to time dimension only by taking mean across other dims
        minimal_data = valid_poses_dataset_with_nan.position.mean(
            ["space", "keypoints", "individuals"]
        )
        report = report_nan_values(minimal_data)
        assert "data:" in report
        assert "Individual:" not in report
