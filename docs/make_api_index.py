"""Generate the API index page for all ``movement`` modules."""

import os

# Modules to exclude from the API index
exclude_modules = ["cli_entrypoint"]

# Set the current working directory to the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def make_api_doctree():
    """Create a doctree of all ``movement`` modules."""
    doctree = "\n"

    for root, _, files in os.walk("../movement"):
        # Remove leading "../"
        root = root[3:]
        for file in sorted(files):
            if file.endswith(".py") and not file.startswith("_"):
                # Convert file path to module name
                full = os.path.join(root, file)
                full = full[:-3].replace(os.sep, ".")
                full = (".").join(full.split("."))
                exclude = False
                for exclude_module in exclude_modules:
                    if file.startswith(exclude_module):
                        exclude = True
                        break
                if not exclude:
                    doctree += f"    {full}\n"

    # Get the header
    with open("./source/_templates/api_index_head.rst") as f:
        api_head = f.read()
    # Write file for api doc with header + doctree
    with open("./source/api_index.rst", "w") as f:
        f.write("..\n  This file is auto-generated.\n\n")
        f.write(api_head)
        f.write(doctree)


if __name__ == "__main__":
    make_api_doctree()
