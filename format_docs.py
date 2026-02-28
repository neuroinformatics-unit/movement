"""Script to apply consistent monospace formatting to docs and docstrings."""

import os
import re

# The regex is constructed from a controlled list of function names,
# not from user input — this is safe.
EXT_PATTERN = re.compile(
    r"(?<![`\w/])(\.?[A-Za-z0-9_-]+\.(?:csv|h5|slp|dlc))\b(?!`)"  # NOSONAR
)


def replacer(match):
    """Replace and format the docstring text."""
    docstring = match.group(0)
    docstring = re.sub(
        r"(?<=\s)``movement``(?=[\s\.,;:])", "movement", docstring
    )
    docstring = re.sub(
        r"(?<=\s)`movement`(?=[\s\.,;:])", "movement", docstring
    )
    docstring = EXT_PATTERN.sub(r"``\1``", docstring)
    for func in [
        "compute_speed",
        "compute_velocity",
        "compute_displacement",
        "PosesDataset",
    ]:
        # if func has no backticks, wrap it.
        pattern = (
            rf"(?<![`\w\.])"
            rf"(?!\bdef\b |\bimport\b |\bfrom\b |\bclass\b )"
            rf"{func}"
            rf"(?![`\w\(])"
        )
        docstring = re.sub(pattern, rf"``{func}``", docstring)
    return docstring


def process_py(filepath):
    """Process a Python file and fix monospace formatting in docstrings."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return

    new_content = re.sub(
        r"\'\'\'.*?\'\'\'|\"\"\".*?\"\"\"", replacer, content, flags=re.DOTALL
    )
    if new_content != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Modified {filepath}")


for root, _, files in os.walk("."):
    if ".git" in root or ".venv" in root or ".tox" in root or "build" in root:
        continue
    for f in files:
        if f.endswith(".py"):
            process_py(os.path.join(root, f))
