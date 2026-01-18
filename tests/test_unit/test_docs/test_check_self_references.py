"""Tests for docs/check_self_references.py."""

import sys
from pathlib import Path

# Add docs directory to path so we can import the module
DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
sys.path.insert(0, str(DOCS_DIR))

from check_self_references import (
    BASE_URL,
    find_self_references,
    is_allowed,
    suggest_target,
)


class TestFindSelfReferences:
    """Tests for find_self_references function."""

    def test_detects_full_url(self, tmp_path):
        """Full URLs to movement docs are detected."""
        test_file = tmp_path / "test.md"
        test_file.write_text(
            f"Check out the [docs]({BASE_URL}/latest/user_guide/installation.html)."
        )

        violations = find_self_references(test_file, BASE_URL)

        assert len(violations) == 1
        assert violations[0][0] == 1  # line number
        assert "installation.html" in violations[0][1]

    def test_detects_multiple_urls_same_line(self, tmp_path):
        """Multiple URLs on the same line are all detected."""
        test_file = tmp_path / "test.md"
        test_file.write_text(
            f"See [{BASE_URL}/latest]({BASE_URL}/latest) "
            f"and [{BASE_URL}/latest/examples]({BASE_URL}/latest/examples/index.html)."
        )

        violations = find_self_references(test_file, BASE_URL)

        # Should detect 4 URLs (2 in link text, 2 in href)
        assert len(violations) == 4

    def test_detects_url_with_anchor(self, tmp_path):
        """URLs with anchors are detected."""
        test_file = tmp_path / "test.md"
        test_file.write_text(
            f"See [section]({BASE_URL}/latest/user_guide/gui.html#target-load-video)."
        )

        violations = find_self_references(test_file, BASE_URL)

        assert len(violations) == 1
        assert "#target-load-video" in violations[0][1]

    def test_ignores_different_domain(self, tmp_path):
        """URLs to other domains are not flagged."""
        test_file = tmp_path / "test.md"
        test_file.write_text(
            "See [napari](https://napari.org/stable/) "
            "and [GitHub](https://github.com/neuroinformatics-unit/movement)."
        )

        violations = find_self_references(test_file, BASE_URL)

        assert len(violations) == 0

    def test_returns_correct_line_numbers(self, tmp_path):
        """Line numbers in violations are correct."""
        test_file = tmp_path / "test.md"
        test_file.write_text(
            "# Header\n"
            "\n"
            "Some text.\n"
            "\n"
            f"A link to [{BASE_URL}/latest]({BASE_URL}/latest).\n"
            "\n"
            f"Another link to [{BASE_URL}/latest/api]({BASE_URL}/latest/api_index.html).\n"
        )

        violations = find_self_references(test_file, BASE_URL)

        line_numbers = [v[0] for v in violations]
        assert 5 in line_numbers
        assert 7 in line_numbers


class TestIsAllowed:
    """Tests for is_allowed function."""

    def test_switcher_json_allowed(self):
        """Switcher JSON URL is allowed."""
        url = f"{BASE_URL}/latest/_static/switcher.json"
        assert is_allowed(url) is True

    def test_regular_docs_url_not_allowed(self):
        """Regular documentation URLs are not allowed."""
        url = f"{BASE_URL}/latest/user_guide/installation.html"
        assert is_allowed(url) is False

    def test_base_url_not_allowed(self):
        """Base URL without path is not allowed by default."""
        assert is_allowed(BASE_URL) is False
        assert is_allowed(f"{BASE_URL}/") is False


class TestSuggestTarget:
    """Tests for suggest_target function."""

    def test_installation_url(self):
        """Installation URL maps to target-installation."""
        url = f"{BASE_URL}/latest/user_guide/installation.html"
        assert suggest_target(url) == "target-installation"

    def test_examples_url(self):
        """Examples URL maps to target-examples."""
        url = f"{BASE_URL}/latest/examples/index.html"
        assert suggest_target(url) == "target-examples"

    def test_mission_url(self):
        """Mission URL maps to target-mission."""
        url = f"{BASE_URL}/latest/community/mission-scope.html"
        assert suggest_target(url) == "target-mission"

    def test_roadmaps_url(self):
        """Roadmaps URL maps to target-roadmaps."""
        url = f"{BASE_URL}/latest/community/roadmaps.html"
        assert suggest_target(url) == "target-roadmaps"

    def test_url_with_anchor_strips_anchor(self):
        """URL with anchor still matches the base target."""
        url = f"{BASE_URL}/latest/user_guide/gui.html#some-section"
        assert suggest_target(url) == "target-gui"

    def test_base_url_maps_to_movement(self):
        """Base URL maps to target-movement."""
        assert suggest_target(f"{BASE_URL}/latest") == "target-movement"
        assert suggest_target(f"{BASE_URL}/latest/") == "target-movement"

    def test_unknown_url_returns_none(self):
        """Unknown URLs return None."""
        url = f"{BASE_URL}/latest/some/unknown/path.html"
        assert suggest_target(url) is None


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_clean_file_has_no_violations(self, tmp_path):
        """Files using internal references have no violations."""
        test_file = tmp_path / "clean.md"
        test_file.write_text(
            "# Clean Documentation\n"
            "\n"
            "See the [installation guide](target-installation) for details.\n"
            "Check [examples](target-examples) for usage.\n"
            "Read about our [mission](target-mission).\n"
        )

        violations = find_self_references(test_file, BASE_URL)

        assert len(violations) == 0

    def test_mixed_file_detects_only_violations(self, tmp_path):
        """Files with mixed content only flag violations."""
        test_file = tmp_path / "mixed.md"
        test_file.write_text(
            "# Mixed Content\n"
            "\n"
            "Internal ref: [guide](target-installation)\n"
            f"External ref: [docs]({BASE_URL}/latest/examples/index.html)\n"
            "External other: [napari](https://napari.org/)\n"
        )

        violations = find_self_references(test_file, BASE_URL)

        assert len(violations) == 1
        assert "examples/index.html" in violations[0][1]
