"""Generate the API index and autosummary pages for ``movement`` modules.

This script generates the top-level API index file (``api_index.rst``)
for all modules in the `movement` package, except for those specified
in ``EXCLUDE_MODULES``.
This script also allows "package modules" that aggregate submodules
via their ``__init__.py`` files to be added to the API index.
These modules are specified in ``MANUAL_MODULES`` and will have their
autosummary pages generated.
"""

import importlib
import inspect
import os
import sys
from pathlib import Path

from jinja2 import FileSystemLoader
from jinja2.sandbox import SandboxedEnvironment
from sphinx.ext.autosummary.generate import _underline
from sphinx.util import rst

# Single-file modules to exclude from the API index
EXCLUDE_MODULES = {
    "movement.cli_entrypoint",
    "movement.napari.loader_widgets",
    "movement.napari.meta_widget",
}

# Modules with __init__.py that expose submodules explicitly
MANUAL_MODULES = {"movement.kinematics", "movement.plots", "movement.roi"}

# Configure paths
SCRIPT_DIR = Path(__file__).resolve().parent
MOVEMENT_ROOT = SCRIPT_DIR.parent
SOURCE_PATH = Path("source")
TEMPLATES_PATH = SOURCE_PATH / "_templates"

os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(MOVEMENT_ROOT))


def get_members(module_name):
    """Return all functions and classes in a module."""
    mod = importlib.import_module(module_name)
    functions = []
    classes = []
    for name in getattr(mod, "__all__", dir(mod)):
        obj = getattr(mod, name, None)
        if inspect.isfunction(obj):
            functions.append(f"{name}")
        elif inspect.isclass(obj):
            classes.append(f"{name}")
    return functions, classes


def write_autosummary_module_page(module_name, output_path):
    """Generate an .rst file with autosummary listing for the given module."""
    functions, classes = get_members(module_name)
    env = SandboxedEnvironment(loader=FileSystemLoader(TEMPLATES_PATH))
    # Add custom autosummary filters
    env.filters["escape"] = rst.escape
    env.filters["underline"] = _underline
    template = env.get_template("autosummary/module.rst")
    content = template.render(
        fullname=module_name,
        underline="=" * len(module_name),
        classes=classes,
        functions=functions,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)


def make_api_index(module_names):
    """Create a top-level API index file listing the specified modules."""
    doctree_lines = [
        f"    {module_name}" for module_name in sorted(module_names)
    ]
    api_head = (TEMPLATES_PATH / "api_index_head.rst").read_text()
    output_path = SOURCE_PATH / "api_index.rst"
    output_path.write_text(api_head + "\n" + "\n".join(doctree_lines))


if __name__ == "__main__":
    # Gather all modules and their paths
    ALL_MODULE_PATHS = sorted((MOVEMENT_ROOT / "movement").rglob("*.py"))
    module_names = set()
    for path in ALL_MODULE_PATHS:
        module_name = str(
            path.relative_to(MOVEMENT_ROOT).with_suffix("")
        ).replace(os.sep, ".")
        if path.name == "__init__.py":
            parent = module_name.rsplit(".", 1)[0]
            if parent in MANUAL_MODULES:
                module_names.add(parent)
            continue
        module_names.add(module_name)
    # Determine submodules of manual modules to exclude
    MANUAL_MODULE_CHILDREN = {
        name
        for name in module_names
        for parent in MANUAL_MODULES
        if name.startswith(parent + ".")
    }
    # Gather list of all modules to exclude
    ALL_EXCLUDE_MODULES = EXCLUDE_MODULES | MANUAL_MODULE_CHILDREN
    # Generate the API index
    make_api_index(module_names - ALL_EXCLUDE_MODULES)
    # Generate autosummary pages for manual modules
    for module_name in MANUAL_MODULES:
        output_path = SOURCE_PATH / "api" / f"{module_name}.rst"
        write_autosummary_module_page(module_name, output_path)
