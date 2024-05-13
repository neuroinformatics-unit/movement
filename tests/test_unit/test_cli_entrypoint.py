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
        ),  # valid arg
        (
            ["movement", "invalid"],
            pytest.raises(SystemExit),
        ),  # invalid arg
        (["movement"], does_not_raise("usage: movement")),  # empty arg
    ],
)
def test_entrypoint_command(command, expected_exception):
    with (
        patch("sys.argv", command),
        patch("builtins.print") as mock_print,
        expected_exception as e,
    ):
        main()
        printed_message = " ".join(map(str, mock_print.call_args.args))
        assert e in printed_message
