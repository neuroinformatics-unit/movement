import builtins
import subprocess
import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from movement.cli_entrypoint import app

runner = CliRunner()


@pytest.mark.parametrize(
    "args, expected_exit_code, expected_output",
    [
        (["info"], 0, "Platform: "),  # Valid command
        (["invalid"], 2, ""),  # Invalid command exits with code 2
        ([], 0, "movement"),  # No command shows help
    ],
)
def test_entrypoint_command(args, expected_exit_code, expected_output):
    """Test the entrypoint with different commands: 'info', 'invalid', ''."""
    result = runner.invoke(app, args)
    assert result.exit_code == expected_exit_code
    assert expected_output in result.output


original_import = builtins.__import__


def fake_import(name, globals, locals, fromlist, level):
    """Pretend that napari is not installed."""
    if name == "napari":
        raise ImportError("No module named 'napari'")
    return original_import(name, globals, locals, fromlist, level)


def test_info_without_napari_installed():
    """Test the 'movement info' can report that napari is not installed."""
    with patch("builtins.__import__", side_effect=fake_import):
        result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "napari: not installed" in result.output


@pytest.mark.parametrize(
    "run_side_effect, expected_exit_code, expected_output",
    [
        (None, 0, ""),  # No error
        (
            subprocess.CalledProcessError(1, "napari"),
            1,
            "error occurred while",
        ),
    ],
)
def test_launch_command(run_side_effect, expected_exit_code, expected_output):
    """Test the 'launch' command.

    We mock the subprocess.run function to avoid actually launching napari.
    """
    with patch("subprocess.run", side_effect=run_side_effect) as mock_run:
        result = runner.invoke(app, ["launch"])
    # Assert that subprocess.run was called with the correct arguments
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == sys.executable
    assert args[1:] == ["-m", "napari", "-w", "movement"]
    # Assert exit code and output
    assert result.exit_code == expected_exit_code
    assert expected_output in result.output
