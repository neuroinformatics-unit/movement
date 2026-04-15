"""CLI entrypoint for the ``movement`` package."""

import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy as sp
import typer
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

app = typer.Typer(
    name="movement",
    help="A Python toolbox for analysing animal body movements.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context) -> None:
    """Show help when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def info() -> None:
    """Output diagnostic information about the environment."""
    text = (
        f"{ASCII_ART}\n"
        f"     movement: {movement.__version__}\n"
        f"     Python: {platform.python_version()}\n"
        f"     NumPy: {np.__version__}\n"
        f"     SciPy: {sp.__version__}\n"
        f"     xarray: {xr.__version__}\n"
        f"     pandas: {pd.__version__}\n"
    )

    try:
        import napari

        text += f"     napari: {napari.__version__}\n"
    except ImportError:
        text += "     napari: not installed\n"

    text += f"     Platform: {platform.platform()}\n"
    typer.echo(text)


@app.command()
def launch() -> None:
    """Launch the movement plugin in napari."""
    try:
        # Use sys.executable to ensure the correct Python interpreter is used
        subprocess.run(
            [sys.executable, "-m", "napari", "-w", "movement"], check=True
        )
    except subprocess.CalledProcessError as e:
        # if subprocess.run() fails with non-zero exit code
        typer.echo(
            "\nAn error occurred while launching the movement plugin "
            f"for napari:\n  {e}"
        )
        raise typer.Exit(code=1)


def main() -> None:
    """Entrypoint for the CLI."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
