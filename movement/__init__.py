from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("movement")
except PackageNotFoundError:
    # package is not installed
    pass
