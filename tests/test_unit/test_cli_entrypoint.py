import builtins
import subprocess
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pytest

from movement.cli_entrypoint import main


@pytest.mark.parametrize(
    "command, expected_exception",
    [
        (
            ["movement", "info"],
            does_not_raise("Platform: "),
        ),  # Valid arg
        (
            ["movement", "invalid"],
            pytest.raises(SystemExit),
        ),  # Invalid arg
        (["movement"], does_not_raise("usage: movement")),  # Empty arg
    ],
)
def test_entrypoint_command(command, expected_exception):
    """Test the entrypoint with different commands: 'info', 'invalid', ''."""
    with (
        patch("sys.argv", command),
        patch("builtins.print") as mock_print,
        expected_exception as e,
    ):
        main()
        printed_message = " ".join(map(str, mock_print.call_args.args))
        assert e in printed_message


original_import = builtins.__import__


def fake_import(name, globals, locals, fromlist, level):
    """Pretend that napari is not installed."""
    if name == "napari":
        raise ImportError("No module named 'napari'")
    return original_import(name, globals, locals, fromlist, level)


def test_info_without_napari_installed():
    """Test the 'movement info' can report that napari is not installed."""
    with (
        patch("sys.argv", ["movement", "info"]),
        patch("builtins.print") as mock_print,
        patch("builtins.__import__", side_effect=fake_import),
    ):
        main()
        printed_message = " ".join(map(str, mock_print.call_args.args))
        assert "napari: not installed" in printed_message


@pytest.mark.parametrize(
    "run_side_effect, expected_message",
    [
        (None, ""),  # No error
        (subprocess.CalledProcessError(1, "napari"), "error occurred while"),
    ],
)
def test_launch_command(run_side_effect, expected_message, capsys):
    """Test the 'launch' command.

    We mock the subprocess.run function to avoid actually launching napari.
    """
    with (
        patch("sys.argv", ["movement", "launch"]),
        patch("subprocess.run", side_effect=run_side_effect) as mock_run,
    ):
        main()
        # Assert that subprocess.run was called with the correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == sys.executable
        assert args[1:] == ["-m", "napari", "-w", "movement"]
        # Assert that the expected message was printed
        captured = capsys.readouterr()
        assert expected_message in captured.out
