"""Convert admonitions GitHub Flavored Markdown to MyST Markdown."""

import re
from pathlib import Path


def convert_gfm_admonitions_to_myst_snippets(
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
    valid_types = {"note", "tip", "important", "warning", "caution"}
    if exclude is None:
        excluded_types = set()  # Empty set
    else:
        excluded_types = set([s.lower() for s in exclude])  # Lowercase

    print(f"Excluded admonition types: {excluded_types}")

    # Read the input file
    with open(input_path, encoding="utf-8") as f:
        gfm_text = f.read()

    # Regex pattern to match GFM admonitions
    pattern = r"(^> \[!(\w+)\]\n(?:^> .*\n?)*)"
    matches = re.finditer(pattern, gfm_text, re.MULTILINE)

    # List to hold converted admonitions
    admonitions = []

    for match in matches:
        full_block = match.group(0)
        adm_type = match.group(2).lower()
        # Skip invalid or excluded admonition types
        if adm_type not in valid_types or adm_type in excluded_types:
            continue
        # Extract content lines, skipping the admonition type line
        content_lines = []
        for line in full_block.split("\n"):
            if line.startswith("> ") and not line.startswith("> [!"):
                content_lines.append(line[2:].strip())
        content = "\n".join(content_lines).strip()
        # Convert to MyST admonition
        adm_myst = ":::{" + adm_type + "}\n" + content + "\n" + ":::\n"
        # Append to the list
        admonitions.append(adm_myst)

    if admonitions:
        # Write all admonitions to a single file
        with open(output_path, "w", encoding="utf-8") as f:
            for admonition in admonitions:
                f.write(admonition + "\n")
        print(f"Admonitions written to {output_path}")
    else:
        print("No GitHub Markdown admonitions found.")


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
    convert_gfm_admonitions_to_myst_snippets(
        readme_path, target_path, exclude={"note"}
    )
