"""CLI entrypoint for the ``movement`` package."""

import argparse
import platform

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

    args = parser.parse_args()
    if args.command is None:
        help_message = parser.format_help()
        print(help_message)
    else:
        args.func()


def info() -> None:
    """Output diagnostic information."""
    print(
        f"{ASCII_ART}\n"
        f"     movement: {movement.__version__}\n"
        f"     Python: {platform.python_version()}\n"
        f"     NumPy: {np.__version__}\n"
        f"     xarray: {xr.__version__}\n"
        f"     pandas: {pd.__version__}\n"
        f"     Platform: {platform.platform()}\n"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
