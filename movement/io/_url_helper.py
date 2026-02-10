"""Helper utilities for downloading remote files via URL."""

from pathlib import Path
from urllib.parse import urlparse

import pooch

from movement.utils.logging import hide_pooch_hash_logs, logger

#: Default cache directory for files downloaded from public URLs.
PUBLIC_DATA_DIR = Path(
    "~", ".movement", "data", "public_datasets"
).expanduser()


def _resolve_url(url: str, cache_dir: Path = PUBLIC_DATA_DIR) -> Path:
    """Download a file from a URL and return the local path.

    Only HTTPS URLs are supported for security. The downloaded file
    is cached locally so subsequent calls with the same URL skip
    the download.

    Parameters
    ----------
    url : str
        The HTTPS URL to download from.
    cache_dir : pathlib.Path, optional
        The directory to cache downloaded files in.

    Returns
    -------
    pathlib.Path
        The path to the locally cached file.

    Raises
    ------
    ValueError
        If the URL does not use the HTTPS scheme.

    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise logger.error(
            ValueError(
                f"Only HTTPS URLs are supported, got: {parsed.scheme}://. "
                "Please provide a URL starting with https://."
            )
        )
    with hide_pooch_hash_logs():
        local_path = pooch.retrieve(
            url=url,
            known_hash=None,
            path=cache_dir,
            progressbar=True,
        )
    return Path(local_path)
