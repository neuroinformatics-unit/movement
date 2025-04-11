from importlib.metadata import PackageNotFoundError, version

from movement.utils.logging import logger

try:
    __version__ = version("movement")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass

# set xarray global options
import xarray as xr

xr.set_options(keep_attrs=True, display_expand_data=False)

# Configure logging to stderr and a file
logger.configure()
