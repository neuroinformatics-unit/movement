from importlib.metadata import PackageNotFoundError, version

from movement.logging import configure_logging

try:
    __version__ = version("movement")
except PackageNotFoundError:
    # package is not installed
    pass

# set xarray global options
import xarray as xr

xr.set_options(keep_attrs=True, display_expand_data=False)

# initialize logger upon import
configure_logging()
