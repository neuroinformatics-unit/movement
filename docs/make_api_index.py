"""Generate the API index page for all ``movement`` modules."""

import importlib
import inspect
import os
import sys
from pathlib import Path

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

    title = module_name
    underline = "=" * len(title)

    lines = [
        f"{title}",
        f"{underline}",
        "",
        ".. rubric:: Description",
        "",
        f".. automodule:: {module_name}",
        "",
        f".. currentmodule:: {module_name}",
        "",
    ]

    if classes:
        lines += [
            ".. rubric:: Classes",
            "",
            ".. autosummary::",
            "    :toctree: .",
            "    :nosignatures:",
            "",
        ]
        for cls in classes:
            lines.append(f"    {cls}")

        lines.append("")

    if functions:
        lines += [
            ".. rubric:: Functions",
            "",
            ".. autosummary::",
            "    :toctree: .",
            "    :nosignatures:",
            "",
        ]
        for func in functions:
            lines.append(f"    {func}")

        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


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
    api_head_path = Path("source") / "_templates" / "api_index_head.rst"
    api_head = api_head_path.read_text()
    # Write api_index.rst with header + doctree
    output_path = Path("source") / "api_index.rst"
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
        output_path = Path("source") / "api" / f"{module_name}.rst"
        write_autosummary_module_page(module_name, output_path)
    # Generate the API index
    make_api_index()
