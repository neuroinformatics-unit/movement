"""Check for external self-references that should use internal cross-references.

This script scans documentation files for URLs pointing to the project's own
documentation site (movement.neuroinformatics.dev) and suggests using internal
MyST cross-references instead. This prevents circular dependency issues when
the documentation site is offline.

Usage:
    python check_self_references.py

Returns exit code 1 if violations are found, 0 otherwise.
"""

import re
import sys
from pathlib import Path

# Configuration
BASE_URL = "https://movement.neuroinformatics.dev"
DOCS_DIR = Path("source")

# URLs to allow (won't trigger errors)
ALLOWLIST_PATTERNS = [
    r"/_static/switcher\.json$",  # Version switcher JSON
]

# Mapping of URL paths to internal target names
URL_TO_TARGET = {
    "/latest": "target-movement",
    "/latest/": "target-movement",
    "/latest/user_guide/installation.html": "target-installation",
    "/latest/user_guide/input_output.html": "target-io",
    "/latest/user_guide/gui.html": "target-gui",
    "/latest/user_guide/movement_dataset.html": "target-poses-and-bboxes-dataset",
    "/latest/examples/index.html": "target-examples",
    "/latest/community/index.html": "target-connect-with-us",
    "/latest/community/mission-scope.html": "target-mission",
    "/latest/community/roadmaps.html": "target-roadmaps",
    "/latest/community/people.html": "target-people",
    "/latest/community/resources.html": "target-resources",
    "/latest/community/contributing.html": "target-contributing",
    "/latest/api_index.html": "target-api",
}

# File extensions to check
FILE_EXTENSIONS = {".md", ".rst"}


def find_self_references(file_path: Path, base_url: str) -> list[tuple[int, str]]:
    """Find external self-references in a file.

    Parameters
    ----------
    file_path : Path
        Path to the file to check.
    base_url : str
        Base URL of the documentation site.

    Returns
    -------
    list[tuple[int, str]]
        List of (line_number, url) tuples for each self-reference found.

    """
    content = file_path.read_text(encoding="utf-8")
    pattern = re.escape(base_url) + r"[^\s\)\]\"'>]*"

    violations = []
    for i, line in enumerate(content.splitlines(), start=1):
        for match in re.finditer(pattern, line):
            url = match.group(0)
            if not is_allowed(url):
                violations.append((i, url))

    return violations


def is_allowed(url: str) -> bool:
    """Check if a URL is in the allowlist.

    Parameters
    ----------
    url : str
        URL to check.

    Returns
    -------
    bool
        True if the URL is allowed, False otherwise.

    """
    for pattern in ALLOWLIST_PATTERNS:
        if re.search(pattern, url):
            return True
    return False


def suggest_target(url: str) -> str | None:
    """Suggest an internal target for a URL.

    Parameters
    ----------
    url : str
        URL to find a target for.

    Returns
    -------
    str | None
        Internal target name if found, None otherwise.

    """
    # Extract path from URL
    path = url.replace(BASE_URL, "")

    # Remove anchor if present
    path = path.split("#")[0]

    # Try exact match first
    if path in URL_TO_TARGET:
        return URL_TO_TARGET[path]

    # Try without trailing slash
    if path.endswith("/") and path[:-1] in URL_TO_TARGET:
        return URL_TO_TARGET[path[:-1]]

    # Try with trailing slash
    if not path.endswith("/") and path + "/" in URL_TO_TARGET:
        return URL_TO_TARGET[path + "/"]

    return None


def check_files(docs_dir: Path) -> list[tuple[Path, int, str, str | None]]:
    """Check all documentation files for self-references.

    Parameters
    ----------
    docs_dir : Path
        Root directory of documentation source files.

    Returns
    -------
    list[tuple[Path, int, str, str | None]]
        List of (file, line_number, url, suggested_target) tuples.

    """
    violations = []

    for ext in FILE_EXTENSIONS:
        for file_path in docs_dir.rglob(f"*{ext}"):
            for line_num, url in find_self_references(file_path, BASE_URL):
                target = suggest_target(url)
                violations.append((file_path, line_num, url, target))

    return violations


def format_output(
    violations: list[tuple[Path, int, str, str | None]], docs_dir: Path
) -> str:
    """Format violations for display.

    Parameters
    ----------
    violations : list[tuple[Path, int, str, str | None]]
        List of (file, line_number, url, suggested_target) tuples.
    docs_dir : Path
        Root directory used to compute relative paths.

    Returns
    -------
    str
        Formatted output string.

    """
    if not violations:
        return "No self-reference violations found."

    lines = ["Self-reference violations found:", ""]

    for file_path, line_num, url, target in violations:
        rel_path = file_path.relative_to(docs_dir.parent)
        lines.append(f"{rel_path}:{line_num}: External URL should use internal ref")
        lines.append(f"  URL: {url}")
        if target:
            lines.append(f"  Use: [link text]({target})")
        else:
            lines.append("  (No matching internal target found)")
        lines.append("")

    lines.append(
        f"Found {len(violations)} self-reference violation(s). "
        "Use internal cross-references instead."
    )

    return "\n".join(lines)


def main() -> int:
    """Run the self-reference checker.

    Returns
    -------
    int
        Exit code: 0 if no violations, 1 if violations found.

    """
    # Determine the docs directory relative to the script
    script_dir = Path(__file__).resolve().parent
    docs_dir = script_dir / DOCS_DIR

    if not docs_dir.exists():
        print(f"Error: Documentation directory not found: {docs_dir}")
        return 1

    violations = check_files(docs_dir)
    output = format_output(violations, docs_dir)
    print(output)

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
