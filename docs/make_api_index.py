"""Generate the API index page for all ``movement`` modules."""

import os
from pathlib import Path

# Modules to exclude from the API index
exclude_modules = [
    "cli_entrypoint",
    "loader_widgets",
    "meta_widget",
]

# Set the current working directory to the directory of this script
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)


def make_api_index():
    """Create a doctree of all ``movement`` modules."""
    doctree = "\n"
    api_path = Path("../movement")
    for path in sorted(api_path.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        # Convert file path to module name
        rel_path = path.relative_to(api_path.parent)
        module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
        if rel_path.stem not in exclude_modules:
            doctree += f"    {module_name}\n"
    # Get the header
    api_head_path = Path("source") / "_templates" / "api_index_head.rst"
    api_head = api_head_path.read_text()
    # Write api_index.rst with header + doctree
    output_path = Path("source") / "api_index.rst"
    with output_path.open("w") as f:
        f.write(api_head)
        f.write(doctree)


if __name__ == "__main__":
    make_api_index()
