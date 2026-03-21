import re
from pathlib import Path
VALID_TYPES = {"note", "tip", "important", "warning", "caution"}
def convert_gfm_admonitions_to_myst_md(
    input_path: Path,
    output_path: Path,
    exclude: set[str] | None = None,
    extra_types: set[str] | None = None,
    append: bool = False,
    verbose: bool = False
    """Convert GFM admonitions to MyST Markdown, with optional enhancements."""
    valid_types = VALID_TYPES.union({t.lower() for t in (extra_types or set())})
    excluded_types = {s.lower() for s in (exclude or set())}
    gfm_text = input_path.read_text(encoding="utf-8")
    pattern = r"(^> \[!(\w+)\]\n(?:^> .*\n?)*)"
    matches = re.finditer(pattern, gfm_text, re.MULTILINE)
    admonitions = []
    for match in matches:
        adm_myst = _process_match(match, valid_types, excluded_types, verbose)
        if adm_myst:
            admonitions.append(adm_myst)
    if admonitions:
        mode = "a" if append else "w"
        with output_path.open(mode, encoding="utf-8") as f:
            f.write("\n".join(admonitions) + "\n")
        print(f"Admonitions written to {output_path} (append={append})")
    else:
        print("No GitHub Markdown admonitions found.")
def _process_match(match: re.Match, valid_types: set[str], excluded_types: set[str], verbose: bool) -> str | None:
    """Process a regex match and return converted admonition if valid."""
    adm_type = match.group(2).lower()
    if adm_type not in valid_types:
        if verbose:
            print(f"Skipped unknown admonition type: {adm_type}")
        return None
    if adm_type in excluded_types:
        if verbose:
            print(f"Skipped excluded admonition type: {adm_type}")
        return None
    full_block = match.group(0)
    content = "\n".join(
        line[2:].strip()
        for line in full_block.split("\n")
        if line.startswith("> ") and not line.startswith("> [!")
    ).strip()
    return f":::{adm_type}\n{content}\n:::\n"
if __name__ == "__main__":
    docs_dir = Path(__file__).resolve().parent
    readme_path = docs_dir.parent / "README.md"
    snippets_dir = docs_dir / "source" / "snippets"
    target_path = snippets_dir / "admonitions.md"
    convert_gfm_admonitions_to_myst_md(
        readme_path,
        target_path,
        exclude={"note"},
        extra_types={"hint"},
        append=False,
        verbose=True
    )
