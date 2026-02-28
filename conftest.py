"""Root-level conftest for doctest collection.

Excludes modules that trigger network calls on import or require
optional dependencies from doctest collection, so that
``--doctest-modules`` works in all environments.
"""

import os

# sample_data.py downloads metadata from the network on import,
# which would fail in fresh CI environments without a cache.
collect_ignore = [os.path.join("movement", "sample_data.py")]

# napari is an optional dependency that may not be installed.
collect_ignore_glob = [os.path.join("movement", "napari", "*.py")]
