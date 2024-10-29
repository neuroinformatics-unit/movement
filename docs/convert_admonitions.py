"""Convert admonitions GitHub Flavored Markdown (GFM) to MyST Markdown."""

import re
from pathlib import Path

# Valid admonition types supported by both GFM and MyST (case-insensitive)
VALID_TYPES = {"note", "tip", "important", "warning", "caution"}


def convert_gfm_admonitions_to_myst_md(
    input_path: Path, output_path: Path, exclude: set[str] | None = None
):
    """Convert admonitions from GitHub Flavored Markdown to MyST.

    Extracts GitHub Flavored Markdown admonitions from the input file and
    writes them to the output file as MyST Markdown admonitions.
    The original admonition type and order are preserved.

    Parameters
    ----------
    input_path : Path
        Path to the input file containing GitHub Flavored Markdown.
    output_path : Path
        Path to the output file to write the MyST Markdown admonitions.
    exclude : set[str], optional
        Set of admonition types to exclude from conversion (case-insensitive).
        Default is None.

    """
    excluded_types = {s.lower() for s in (exclude or set())}

    # Read the input file
    gfm_text = input_path.read_text(encoding="utf-8")

    # Regex pattern to match GFM admonitions
    pattern = r"(^> \[!(\w+)\]\n(?:^> .*\n?)*)"
    matches = re.finditer(pattern, gfm_text, re.MULTILINE)

    # Process matches and collect converted admonitions
    admonitions = []
    for match in matches:
        adm_myst = _process_match(match, excluded_types)
        if adm_myst:
            admonitions.append(adm_myst)

    if admonitions:
        # Write all admonitions to a single file
        output_path.write_text("\n".join(admonitions) + "\n", encoding="utf-8")
        print(f"Admonitions written to {output_path}")
    else:
        print("No GitHub Markdown admonitions found.")


def _process_match(match: re.Match, excluded_types: set[str]) -> str | None:
    """Process a regex match and return the converted admonition if valid."""
    # Extract the admonition type
    adm_type = match.group(2).lower()
    if adm_type not in VALID_TYPES or adm_type in excluded_types:
        return None

    # Extract the content lines
    full_block = match.group(0)
    content = "\n".join(
        line[2:].strip()
        for line in full_block.split("\n")
        if line.startswith("> ") and not line.startswith("> [!")
    ).strip()

    # Return the converted admonition
    return ":::{" + adm_type + "}\n" + content + "\n" + ":::\n"


if __name__ == "__main__":
    # Path to the README.md file
    # (1 level above the current script)
    docs_dir = Path(__file__).resolve().parent
    readme_path = docs_dir.parent / "README.md"

    # Path to the output file
    # (inside the docs/source/snippets directory)
    snippets_dir = docs_dir / "source" / "snippets"
    target_path = snippets_dir / "admonitions.md"

    # Call the function
    convert_gfm_admonitions_to_myst_md(
        readme_path, target_path, exclude={"note"}
    )
