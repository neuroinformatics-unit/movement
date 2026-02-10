"""Test suite for the _url_helper module."""

from unittest.mock import patch

import pytest

from movement.io._url_helper import _resolve_url


def test_resolve_url_https(tmp_path):
    """Test that _resolve_url downloads a file for an HTTPS URL."""
    fake_local = tmp_path / "downloaded.csv"
    fake_local.touch()
    url = "https://example.com/data/file.csv"
    with patch("movement.io._url_helper.pooch.retrieve") as mock_retrieve:
        mock_retrieve.return_value = str(fake_local)
        result = _resolve_url(url, cache_dir=tmp_path)
        mock_retrieve.assert_called_once()
        call_kwargs = mock_retrieve.call_args
        assert call_kwargs.kwargs["url"] == url
        assert call_kwargs.kwargs["known_hash"] is None
        assert result == fake_local


def test_resolve_url_rejects_http():
    """Test that _resolve_url raises ValueError for HTTP URLs."""
    with pytest.raises(ValueError, match="Only HTTPS URLs are supported"):
        _resolve_url("http://example.com/data/file.csv")


def test_resolve_url_rejects_ftp():
    """Test that _resolve_url raises ValueError for non-HTTPS schemes."""
    with pytest.raises(ValueError, match="Only HTTPS URLs are supported"):
        _resolve_url("ftp://example.com/data/file.csv")
