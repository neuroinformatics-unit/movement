"""Generate the API index page for all ``movement`` modules."""

import importlib
import inspect
import os
import sys
from pathlib import Path

from jinja2 import FileSystemLoader
from jinja2.sandbox import SandboxedEnvironment
from sphinx.ext.autosummary.generate import _underline
from sphinx.util import rst

# Modules to exclude from the API index
exclude_modules = [
    "movement.cli_entrypoint",
    "movement.napari.loader_widgets",
    "movement.napari.meta_widget",
]

# Modules with submodules imported in __init__.py
manual_modules = ["movement.kinematics", "movement.plots", "movement.roi"]

# Set the current working directory to the directory of this script
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
movement_root = script_dir.parent  # Go up one level from docs/ to movement/
sys.path.insert(0, str(movement_root))
source_path = Path("source")
templates_path = source_path / "_templates"


def get_members(module_name):
    """Return all functions and classes in a module."""
    mod = importlib.import_module(module_name)
    functions = []
    classes = []
    for name in getattr(mod, "__all__", dir(mod)):
        obj = getattr(mod, name, None)
        if obj is None:
            continue
        if inspect.isfunction(obj):
            functions.append(f"{name}")
        elif inspect.isclass(obj):
            classes.append(f"{name}")
    return functions, classes


def write_autosummary_module_page(module_name, output_path):
    """Generate an .rst file with autosummary listing for the given module."""
    functions, classes = get_members(module_name)
    env = SandboxedEnvironment(loader=FileSystemLoader(templates_path))
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


def make_api_index():
    """Create a doctree of all ``movement`` modules."""
    doctree = "\n"
    api_path = movement_root / "movement"
    for path in sorted(api_path.rglob("*.py")):
        is_init_file = path.name == "__init__.py"
        module_name = str(
            path.relative_to(movement_root).with_suffix("")
        ).replace(os.sep, ".")
        is_manual_module = any(
            module_name.startswith(mod) for mod in manual_modules
        )
        if path.name.startswith("_") and not is_manual_module:
            continue
        if is_init_file:
            # Strip "__init__" from module name
            module_name = module_name.rsplit(".", 1)[0]
        if module_name not in exclude_modules:
            doctree += f"    {module_name}\n"
    # Get the header
    api_head_path = templates_path / "api_index_head.rst"
    api_head = api_head_path.read_text()
    # Write api_index.rst with header + doctree
    output_path = source_path / "api_index.rst"
    with output_path.open("w") as f:
        f.write(api_head)
        f.write(doctree)


if __name__ == "__main__":
    # Generate autosummary pages for manual modules
    for module_name in manual_modules:
        mod_path = movement_root / Path(module_name.replace(".", os.sep))
        if mod_path.exists():
            for submodule in mod_path.rglob("*.py"):
                if submodule.name != "__init__.py":
                    # Add submodule to exclude list
                    exclude_modules.append(
                        str(
                            submodule.relative_to(movement_root).with_suffix(
                                ""
                            )
                        ).replace(os.sep, ".")
                    )
        output_path = source_path / "api" / f"{module_name}.rst"
        write_autosummary_module_page(module_name, output_path)
    # Generate the API index
    make_api_index()
