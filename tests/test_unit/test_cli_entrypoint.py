from unittest.mock import patch

import pytest

from movement.cli_entrypoint import main


@pytest.mark.parametrize(
    "command, expected_substring",
    [("info", "movement: "), ("invalid arg", "Invalid command.")],
)
def test_entrypoint_command(command, expected_substring):
    with (
        patch("sys.argv", ["cli_entrypoint", command]),
        patch("builtins.print") as mock_print,
    ):
        main()
        printed_message = " ".join(map(str, mock_print.call_args.args))
        assert expected_substring in printed_message
