# test_body_axis.py
"""Tests for the body axis validation module."""

from typing import Any

import pytest

from movement.kinematics.body_axis import ValidateAPConfig


class TestValidateAPConfig:
    """Tests for the ValidateAPConfig dataclass parameter validation."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("min_valid_frac", -0.1),
            ("min_valid_frac", 1.1),
            ("window_len", 0),
            ("window_len", -5),
            ("window_len", 2.5),
            ("stride", 0),
            ("stride", -1),
            ("stride", 1.5),
            ("pct_thresh", -1),
            ("pct_thresh", 101),
            ("min_run_len", 0),
            ("min_run_len", -1),
            ("min_run_len", 1.5),
            ("postural_var_ratio_thresh", 0),
            ("postural_var_ratio_thresh", -1),
            ("max_clusters", 0),
            ("max_clusters", 2.5),
            ("confidence_floor", -0.1),
            ("confidence_floor", 1.1),
            ("lateral_thresh_pct", -1),
            ("lateral_thresh_pct", 101),
            ("edge_thresh_pct", -1),
            ("edge_thresh_pct", 101),
            ("lateral_var_weight", -0.1),
            ("longitudinal_var_weight", -0.1),
        ],
    )
    def test_invalid_config_values_raise(self, field: str, value: Any) -> None:
        """Invalid config values should raise ValueError."""
        kwargs = {field: value}
        with pytest.raises(ValueError, match="must be"):
            ValidateAPConfig(**kwargs)

    def test_valid_config_does_not_raise(self) -> None:
        """Valid config values should not raise any error."""
        # Should not raise
        ValidateAPConfig(
            min_valid_frac=0.5,
            window_len=10,
            stride=2,
            pct_thresh=50.0,
            min_run_len=2,
            postural_var_ratio_thresh=1.5,
            max_clusters=3,
            confidence_floor=0.2,
            lateral_thresh_pct=50.0,
            edge_thresh_pct=70.0,
            lateral_var_weight=1.0,
            longitudinal_var_weight=0.0,
        )
