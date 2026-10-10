"""Report numpydoc sections that appear out of order in package docstrings."""

import ast
import sys
import tokenize
from pathlib import Path

# See https://numpydoc.readthedocs.io/en/latest/format.html#sections
SECTION_ORDER = (
    "Parameters",
    "Attributes",
    "Methods",
    "Returns",
    "Yields",
    "Receives",
    "Other Parameters",
    "Raises",
    "Warns",
    "Warnings",
    "See Also",
    "Notes",
    "References",
    "Examples",
)
SECTION_RANK = {section: rank for rank, section in enumerate(SECTION_ORDER)}


def _headings(docstring: str):
    """Yield section names and line offsets for genuine numpydoc headings."""
    lines = docstring.splitlines()
    indents = [
        len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()
    ]
    base_indent = min(indents, default=0)

    for index in range(1, len(lines) - 1):
        line = lines[index]
        name = line.strip()
        if name not in SECTION_RANK or lines[index - 1].strip():
            continue
        if len(line) - len(line.lstrip()) != base_indent:
            continue
        underline = lines[index + 1]
        if len(underline) - len(underline.lstrip()) != base_indent:
            continue
        if underline.strip() == "-" * len(name):
            yield name, index


def check_file(path: Path) -> list[str]:
    """Return section-order errors in a Python file."""
    with tokenize.open(path) as source_file:
        tree = ast.parse(source_file.read(), filename=str(path))

    errors = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        ):
            continue
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue
        expression = node.body[0].value
        if not isinstance(expression, ast.Constant) or not isinstance(
            expression.value, str
        ):
            continue

        latest_name = None
        latest_rank = -1
        for name, offset in _headings(expression.value):
            rank = SECTION_RANK[name]
            if rank < latest_rank:
                errors.append(
                    f"{path}:{expression.lineno + offset}: "
                    f"'{name}' must precede '{latest_name}'"
                )
            else:
                latest_name = name
                latest_rank = rank
    return errors


def main(paths: list[str]) -> int:
    """Print section-order errors and return a pre-commit exit status."""
    errors = []
    for path in paths:
        errors.extend(check_file(Path(path)))
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
