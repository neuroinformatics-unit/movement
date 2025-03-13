"""CLI entrypoint for the ``movement`` package."""

import argparse
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr

import movement

# ASCII Art for branding in CLI output
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
    # Create argument parser to handle commands from the CLI
    parser = argparse.ArgumentParser(prog="movement")
    subparsers = parser.add_subparsers(dest="command", title="commands")

    # Add 'info' command for diagnostic information
    info_parser = subparsers.add_parser(
        "info", help="output diagnostic information about the environment"
    )
    info_parser.set_defaults(func=info)  # Associate the 'info' function with the 'info' command

    # Add 'launch' command to launch the movement plugin in napari
    launch_parser = subparsers.add_parser(
        "launch", help="launch the movement plugin in napari"
    )
    launch_parser.set_defaults(func=launch)  # Associate the 'launch' function with the 'launch' command

    # Parse the arguments from the CLI
    args = parser.parse_args()
    if args.command is None:
        # If no command is specified, print the help message
        help_message = parser.format_help()
        print(help_message)
    else:
        # Call the appropriate function based on the selected command
        args.func()


def info() -> None:
    """Output diagnostic information about the environment."""
    text = (
        f"{ASCII_ART}\n"  # Display the ASCII art logo
        f"     movement: {movement.__version__}\n"  # Show version of the movement package
        f"     Python: {platform.python_version()}\n"  # Show Python version
        f"     NumPy: {np.__version__}\n"  # Show NumPy version
        f"     xarray: {xr.__version__}\n"  # Show xarray version
        f"     pandas: {pd.__version__}\n"  # Show pandas version
    )

    try:
        import napari  # Try importing napari
        text += f"     napari: {napari.__version__}\n"  # Show napari version if installed
    except ImportError:
        # If napari is not installed, indicate that it's missing
        text += "     napari: not installed\n"

    text += f"     Platform: {platform.platform()}\n"  # Show platform details
    print(text)


def launch() -> None:
    """Launch the movement plugin in napari."""
    try:
        # Use sys.executable to ensure the correct Python interpreter is used
        subprocess.run(
            [sys.executable, "-m", "napari", "-w", "movement"],  # Command to run napari with the movement plugin
            check=True,  # This ensures that an exception is raised if the subprocess fails
            stdout=sys.stdout,  # Redirect the standard output to the terminal
            stderr=sys.stderr   # Redirect the error output to the terminal
        )
    except subprocess.CalledProcessError as e:
        # Detailed exception handling for subprocess errors
        print(f"\nError occurred while launching the movement plugin for napari:\n")
        print(f"Exit Code: {e.returncode}")  # Print the return code of the failed subprocess
        print(f"Error Output: {e.stderr.decode() if e.stderr else 'No error output'}")  # Print any stderr output
        print(f"Command: {e.cmd}")  # Print the command that was run for context
    except Exception as e:
        # Catch any other unexpected exceptions and print them
        print(f"\nUnexpected error: {str(e)}")


if __name__ == "__main__":  # pragma: no cover
    # Entry point of the script to start the CLI when executed
    main()

