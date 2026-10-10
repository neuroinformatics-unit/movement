"""Tests for the numpydoc section-order pre-commit hook."""

from pathlib import Path

from tools.check_docstring_sections import check_file


def test_valid_section_order(tmp_path: Path):
    """Accept canonical order in a function docstring."""
    source = tmp_path / "valid.py"
    source.write_text(
        "def outer():\n"
        '    """Summary.\n\n'
        "    Parameters\n"
        "    ----------\n"
        "    value : int\n"
        "        Input.\n\n"
        "    See Also\n"
        "    --------\n"
        "    other\n\n"
        "    Notes\n"
        "    -----\n"
        "    More detail.\n"
        '    """\n'
    )

    assert check_file(source) == []


def test_invalid_section_order(tmp_path: Path):
    """Report a valid heading appearing after a later section."""
    source = tmp_path / "invalid.py"
    source.write_text(
        "def outer():\n"
        '    """Summary.\n\n'
        "    Notes\n"
        "    -----\n"
        "    More detail.\n\n"
        "    See Also\n"
        "    --------\n"
        "    other\n"
        '    """\n'
    )

    assert check_file(source) == [
        f"{source}:8: 'See Also' must precede 'Notes'"
    ]


def test_heading_like_prose_is_ignored(tmp_path: Path):
    """Ignore prose, short underlines, and headings nested in examples."""
    source = tmp_path / "prose.py"
    source.write_text(
        "def outer():\n"
        '    """Summary.\n\n'
        "    Notes\n"
        "    -----\n"
        "    This mentions See Also in ordinary prose.\n"
        "    See Also\n"
        "    --------\n"
        "    Not a section without a blank line.\n\n"
        "    Examples\n"
        "    --------\n"
        "        See Also\n"
        "        --------\n"
        '    """\n'
    )

    assert check_file(source) == []
