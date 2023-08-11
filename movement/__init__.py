from importlib.metadata import PackageNotFoundError, version
from movement.logging import configure_logging

try:
    __version__ = version("movement")
except PackageNotFoundError:
    # package is not installed
    pass


# initialize logger upon import
configure_logging()
