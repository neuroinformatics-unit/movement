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


# initialize logger upon import
configure_logging()

# Import public datasets module functions to make them available at package level
from movement.public_data import list_public_datasets, get_dataset_info

# Configure logging to stderr and a file
logger.configure()
