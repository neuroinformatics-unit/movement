"""CLI entrypoint for the ``movement`` package."""

import argparse
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr

import movement

ASCII_ART = r"""
      _ __ ___   _____   _____ _ __ ___   ___ _ __ | |_
     | '_ ` _ \ / _ \ \ / / _ \ '_ ` _ \ / _ \ '_ \| __|
     | | | | | | (_) \ V /  __/ | | | | |  __/ | | | |_
     |_| |_| |_|\___/ \_/ \___|_| |_| |_|\___|_| |_|\__|

                       .******
                   ,******/*******
               ,*******/******(###  **
           ,/*************(###((**/****/**
       ,*/******/*****(####(***/***/**/***(#.
     .*******/****((###(****/**/***/**(#####//////
     ...*.****(####(****/**/****/*(#####//////////////
     .,.*...###(*****/*/****/*(#####///////////////###
     ...*.,.#(#,,,/******/(##(##///////////////#######
     ..**...###,,,*,,,(#####///////////////###########
     ...*..*(##,,,*,,,###.////////////(###############
     .../...###/,,*,,,###...,.////(###################
     ...*...(##,,,*/,,###.,.,...####################(#
         ...###,*,*,,,###...,...################(#####
               ,,,*,*,##(..*,...############(######(
                 ,*,,,###...,..*########(######(
                     ,#/ .../...####(######(
                         ,..,...(######(
                             ,..###(
    """


def main() -> None:
    """Entrypoint for the CLI."""
    parser = argparse.ArgumentParser(prog="movement")
    subparsers = parser.add_subparsers(dest="command", title="commands")

    # Add 'info' command
    info_parser = subparsers.add_parser(
        "info", help="output diagnostic information about the environment"
    )
    info_parser.set_defaults(func=info)

    # Add 'launch' command
    launch_parser = subparsers.add_parser(
        "launch", help="launch the movement plugin in napari"
    )
    launch_parser.set_defaults(func=launch)

    args = parser.parse_args()
    if args.command is None:
        help_message = parser.format_help()
        print(help_message)
    else:
        args.func()


def info() -> None:
    """Output diagnostic information."""
    text = (
        f"{ASCII_ART}\n"
        f"     movement: {movement.__version__}\n"
        f"     Python: {platform.python_version()}\n"
        f"     NumPy: {np.__version__}\n"
        f"     xarray: {xr.__version__}\n"
        f"     pandas: {pd.__version__}\n"
    )

    try:
        import napari

        text += f"     napari: {napari.__version__}\n"
    except ImportError:
        text += "     napari: not installed\n"

    text += f"     Platform: {platform.platform()}\n"
    print(text)


def launch() -> None:
    """Launch the movement plugin in napari."""
    try:
        # Use sys.executable to ensure the correct Python interpreter is used
        subprocess.run(
            [sys.executable, "-m", "napari", "-w", "movement"], check=True
        )
    except subprocess.CalledProcessError as e:
        # if subprocess.run() fails with non-zero exit code
        print(
            "\nAn error occurred while launching the movement plugin "
            f"for napari:\n  {e}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
