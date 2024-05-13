"""CLI entrypoint for the movement package."""

import platform
import sys

import numpy as np
import pandas as pd
import xarray as xr

import movement

ASCII_ART = """
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
    if len(sys.argv) != 2 or sys.argv[1] != "info":
        print(
            "Invalid command.",
            "Please use 'movement info' to get information.",
        )
        return
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
