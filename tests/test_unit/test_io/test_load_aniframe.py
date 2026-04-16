"""Test suite for the load_aniframe module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr

from movement.io.load_aniframe import (
    _check_no_polar_coords,
    _flatten_r_metadata,
    _r_value_to_python,
    _resolve_columns,
    _resolve_fps,
    from_aniframe_file,
)
from movement.validators.datasets import ValidPosesInputs
from movement.validators.files import ValidAniframeParquet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal bytes that satisfy the 'b"r"' key check in the validator.
# These bytes are not valid R serialisation, so tests that exercise the full
# metadata-decoding path must mock ``_decode_aniframe_metadata`` instead.
_FAKE_R_META = b"A\n3\n263425\n197888\n5\nUTF-8\n"

_BASE_META: dict = {
    "source": "SLEAP",
    "sampling_rate": 30.0,
    "unit_time": "frame",
    "unit_space": "px",
    "reference_frame": "allocentric",
    "point_of_reference": "top_left",
}


def _make_parquet(
    tmp_path: Path,
    *,
    df: pd.DataFrame | None = None,
    r_meta_bytes: bytes | None = _FAKE_R_META,
    filename: str = "test.parquet",
) -> Path:
    """Write a Parquet file with optional aniframe-style schema metadata.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    df
        DataFrame to write. Defaults to a minimal two-individual dataset.
    r_meta_bytes
        Bytes to store under the ``b"r"`` schema metadata key.
        Pass ``None`` to omit the key entirely (invalid aniframe file).
    filename
        Output file name.

    Returns
    -------
    Path
        Path to the written Parquet file.

    """
    if df is None:
        df = _minimal_df()
    table = pa.Table.from_pandas(df, preserve_index=False)
    if r_meta_bytes is not None:
        existing = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing, b"r": r_meta_bytes})
    path = tmp_path / filename
    pq.write_table(table, path)
    return path


def _minimal_df(
    *,
    individuals: list[str | int] | None = None,
    keypoints: list[str] | None = None,
    n_frames: int = 5,
    include_confidence: bool = True,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Build a minimal long-format aniframe-style DataFrame.

    Parameters
    ----------
    individuals
        Individual identifiers. Defaults to ``[1, 2]``.
    keypoints
        Keypoint names. Defaults to ``["nose", "tail"]``.
    n_frames
        Number of time steps.
    include_confidence
        Whether to include a ``confidence`` column.
    extra_cols
        Additional scalar-valued columns to add (e.g. ``{"session": 1}``).

    Returns
    -------
    pandas.DataFrame

    """
    if individuals is None:
        individuals = [1, 2]
    if keypoints is None:
        keypoints = ["nose", "tail"]

    rng = np.random.default_rng(seed=0)
    rows = []
    for ind in individuals:
        for kp in keypoints:
            for t in range(1, n_frames + 1):
                row: dict = {
                    "individual": ind,
                    "keypoint": kp,
                    "time": t,
                    "x": rng.standard_normal(),
                    "y": rng.standard_normal(),
                }
                if include_confidence:
                    row["confidence"] = rng.uniform(0.5, 1.0)
                if extra_cols:
                    row.update(extra_cols)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ValidAniframeParquet tests
# ---------------------------------------------------------------------------


def test_valid_aniframe_parquet_accepts_valid_file(tmp_path):
    """ValidAniframeParquet accepts a Parquet file with the aniframe key."""
    path = _make_parquet(tmp_path)
    v = ValidAniframeParquet(file=path)
    assert v.file == path


def test_valid_aniframe_parquet_rejects_missing_r_key(tmp_path):
    """ValidAniframeParquet rejects a Parquet file without the 'r' key."""
    path = _make_parquet(tmp_path, r_meta_bytes=None)
    with pytest.raises(ValueError, match="aniframe metadata key"):
        ValidAniframeParquet(file=path)


def test_valid_aniframe_parquet_rejects_non_parquet(tmp_path):
    """ValidAniframeParquet rejects files with wrong suffix."""
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="suffix"):
        ValidAniframeParquet(file=path)


def test_valid_aniframe_parquet_rejects_nonexistent_file(tmp_path):
    """ValidAniframeParquet rejects a file that does not exist."""
    path = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        ValidAniframeParquet(file=path)


# ---------------------------------------------------------------------------
# _r_value_to_python tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (float("nan"), None),
        ("SLEAP", "SLEAP"),
        (42, 42),
        (np.array(["SLEAP"], dtype=object), "SLEAP"),
        (np.array([1.0]), 1.0),
        (np.array([float("nan")]), None),
        (
            np.array(["a", "b"], dtype=object),
            ["a", "b"],
        ),
        (pd.Categorical(["frame"]), "frame"),
        (pd.Categorical([]), None),
    ],
)
def test_r_value_to_python(value, expected):
    """_r_value_to_python converts rdata-decoded R values correctly."""
    result = _r_value_to_python(value)
    assert result == expected


# ---------------------------------------------------------------------------
# _flatten_r_metadata tests
# ---------------------------------------------------------------------------


def test_flatten_r_metadata_returns_plain_dict():
    """_flatten_r_metadata converts a dict of R values to plain Python."""
    r_obj = {
        "source": np.array(["SLEAP"], dtype=object),
        "sampling_rate": np.array([30.0]),
        "unit_time": pd.Categorical(["frame"]),
    }
    result = _flatten_r_metadata(r_obj)
    assert result == {
        "source": "SLEAP",
        "sampling_rate": 30.0,
        "unit_time": "frame",
    }


def test_flatten_r_metadata_nested_attributes_metadata():
    """_flatten_r_metadata navigates the real aniframe nested structure."""
    r_obj = {
        "attributes": {
            "metadata": {
                "source": np.array(["SLEAP"], dtype=object),
                "unit_time": pd.Categorical(["frame"]),
            }
        },
        "columns": {"individual": None},
    }
    result = _flatten_r_metadata(r_obj)
    assert result == {"source": "SLEAP", "unit_time": "frame"}


def test_flatten_r_metadata_returns_empty_for_non_dict():
    """_flatten_r_metadata returns an empty dict for non-dict input."""
    assert _flatten_r_metadata("not a dict") == {}
    assert _flatten_r_metadata(None) == {}


# ---------------------------------------------------------------------------
# _resolve_columns tests
# ---------------------------------------------------------------------------


def test_resolve_columns_renames_track_to_individual():
    """_resolve_columns renames 'track' to 'individual' with a UserWarning."""
    df = _minimal_df().rename(columns={"individual": "track"})
    with pytest.warns(UserWarning, match="renamed to 'individual'"):
        result = _resolve_columns(df)
    assert "individual" in result.columns
    assert "track" not in result.columns


def test_resolve_columns_drops_single_value_extra_cols(caplog):
    """_resolve_columns drops constant extra columns at INFO level."""
    df = _minimal_df(extra_cols={"session": 1, "trial": 1})
    result = _resolve_columns(df)
    assert "session" not in result.columns
    assert "trial" not in result.columns


@pytest.mark.parametrize("extra_col", ["session", "trial", "model"])
def test_resolve_columns_errors_on_multi_value_extra_cols(extra_col):
    """_resolve_columns raises ValueError for multi-valued extra columns."""
    df = _minimal_df()
    df[extra_col] = [i % 2 for i in range(len(df))]
    with pytest.raises(ValueError, match="unique values"):
        _resolve_columns(df)


def test_resolve_columns_leaves_canonical_columns_intact():
    """_resolve_columns does not modify individual, keypoint, time, x, y."""
    df = _minimal_df()
    result = _resolve_columns(df)
    for col in ("individual", "keypoint", "time", "x", "y", "confidence"):
        assert col in result.columns


# ---------------------------------------------------------------------------
# _check_no_polar_coords tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("polar_col", ["rho", "phi", "theta"])
def test_check_no_polar_coords_raises_for_polar(polar_col):
    """_check_no_polar_coords raises for polar/spherical column names."""
    with pytest.raises(ValueError, match="Polar"):
        _check_no_polar_coords({"individual", "time", polar_col})


def test_check_no_polar_coords_passes_for_cartesian():
    """_check_no_polar_coords passes when only Cartesian columns exist."""
    _check_no_polar_coords({"individual", "keypoint", "time", "x", "y", "z"})


# ---------------------------------------------------------------------------
# _resolve_fps tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fps_override, unit_time, sampling_rate, expected",
    [
        # Explicit override always wins
        (25.0, "frame", 30.0, 25.0),
        (25.0, "s", 30.0, 25.0),
        # frame units → None regardless of sampling_rate
        (None, "frame", 30.0, None),
        (None, "frame", None, None),
        (None, None, 30.0, None),
        # seconds → use sampling_rate
        (None, "s", 30.0, 30.0),
        # milliseconds → use sampling_rate
        (None, "ms", 50.0, 50.0),
        # sampling_rate missing → None
        (None, "s", None, None),
    ],
)
def test_resolve_fps(fps_override, unit_time, sampling_rate, expected):
    """_resolve_fps returns the correct fps for various metadata."""
    meta = {"unit_time": unit_time, "sampling_rate": sampling_rate}
    result = _resolve_fps(fps_override, meta)
    assert result == expected


# ---------------------------------------------------------------------------
# from_aniframe_file integration tests (metadata is mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def aniframe_parquet(tmp_path):
    """Return a minimal valid aniframe Parquet path (metadata mocked)."""
    return _make_parquet(tmp_path)


def _mock_meta(**overrides):
    """Return a copy of the base metadata with optional overrides."""
    return {**_BASE_META, **overrides}


def test_from_aniframe_file_returns_valid_dataset(aniframe_parquet):
    """from_aniframe_file returns a valid movement poses Dataset."""
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(),
    ):
        ds = from_aniframe_file(aniframe_parquet)

    assert isinstance(ds, xr.Dataset)
    assert set(ValidPosesInputs.DIM_NAMES) == set(ds.dims)
    assert "position" in ds
    assert "confidence" in ds
    assert ds.position.ndim == 4
    assert ds.confidence.ndim == 3


def test_from_aniframe_file_correct_shape(aniframe_parquet):
    """from_aniframe_file produces arrays with the expected shape."""
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(),
    ):
        ds = from_aniframe_file(aniframe_parquet)

    # Minimal df: 2 individuals, 2 keypoints, 5 frames, 2 space dims
    assert ds.sizes["individuals"] == 2
    assert ds.sizes["keypoints"] == 2
    assert ds.sizes["time"] == 5
    assert ds.sizes["space"] == 2
    assert list(ds.coords["space"].values) == ["x", "y"]


def test_from_aniframe_file_source_software_from_metadata(aniframe_parquet):
    """from_aniframe_file sets source_software from aniframe metadata."""
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(source="DeepLabCut"),
    ):
        ds = from_aniframe_file(aniframe_parquet)

    assert ds.attrs["source_software"] == "DeepLabCut"


def test_from_aniframe_file_missing_source_software(
    aniframe_parquet,
):
    """from_aniframe_file sets source_software to None when source is NA."""
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(source=None),
    ):
        ds = from_aniframe_file(aniframe_parquet)

    assert ds.attrs.get("source_software") is None


def test_from_aniframe_file_fps_from_metadata(aniframe_parquet):
    """from_aniframe_file reads fps from sampling_rate when unit_time='s'."""
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(unit_time="s", sampling_rate=25.0),
    ):
        ds = from_aniframe_file(aniframe_parquet)

    assert ds.attrs.get("fps") == pytest.approx(25.0)
    assert ds.attrs.get("time_unit") == "seconds"


def test_from_aniframe_file_fps_override_takes_precedence(aniframe_parquet):
    """from_aniframe_file uses caller-supplied fps over metadata fps."""
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(unit_time="s", sampling_rate=25.0),
    ):
        ds = from_aniframe_file(aniframe_parquet, fps=60.0)

    assert ds.attrs.get("fps") == pytest.approx(60.0)


def test_from_aniframe_file_frame_time_when_unit_is_frame(aniframe_parquet):
    """from_aniframe_file uses frame-number time when unit_time='frame'."""
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(unit_time="frame"),
    ):
        ds = from_aniframe_file(aniframe_parquet)

    assert ds.attrs.get("time_unit") == "frames"
    assert "fps" not in ds.attrs


def test_from_aniframe_file_individual_names_are_strings(aniframe_parquet):
    """from_aniframe_file coerces individual names to strings."""
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(),
    ):
        ds = from_aniframe_file(aniframe_parquet)

    assert all(isinstance(n, str) for n in ds.coords["individuals"].values)


def test_from_aniframe_file_no_confidence_fills_nan(tmp_path):
    """from_aniframe_file fills confidence with NaN when column is absent."""
    df = _minimal_df(include_confidence=False)
    path = _make_parquet(tmp_path, df=df)
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(),
    ):
        ds = from_aniframe_file(path)

    assert np.all(np.isnan(ds.confidence.values))


def test_from_aniframe_file_3d_cartesian(tmp_path):
    """from_aniframe_file handles 3D Cartesian data (x, y, z)."""
    df = _minimal_df()
    rng = np.random.default_rng(seed=1)
    df["z"] = rng.standard_normal(len(df))
    path = _make_parquet(tmp_path, df=df)
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(),
    ):
        ds = from_aniframe_file(path)

    assert ds.sizes["space"] == 3
    assert list(ds.coords["space"].values) == ["x", "y", "z"]


def test_from_aniframe_file_track_renamed_to_individual(tmp_path):
    """from_aniframe_file renames 'track' to 'individual' with a warning."""
    df = _minimal_df().rename(columns={"individual": "track"})
    path = _make_parquet(tmp_path, df=df)
    with (
        patch(
            "movement.io.load_aniframe._decode_aniframe_metadata",
            return_value=_mock_meta(),
        ),
        pytest.warns(UserWarning, match="renamed to 'individual'"),
    ):
        ds = from_aniframe_file(path)

    assert "individuals" in ds.dims


def test_from_aniframe_file_single_value_session_dropped(tmp_path):
    """from_aniframe_file drops a constant session column without error."""
    df = _minimal_df(extra_cols={"session": 1, "trial": 1})
    path = _make_parquet(tmp_path, df=df)
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(),
    ):
        ds = from_aniframe_file(path)

    assert isinstance(ds, xr.Dataset)


@pytest.mark.parametrize("extra_col", ["session", "trial", "model"])
def test_from_aniframe_file_multi_value_extra_col_raises(tmp_path, extra_col):
    """from_aniframe_file raises ValueError for multi-valued extra columns."""
    df = _minimal_df()
    df[extra_col] = [i % 2 for i in range(len(df))]
    path = _make_parquet(tmp_path, df=df)
    with (
        patch(
            "movement.io.load_aniframe._decode_aniframe_metadata",
            return_value=_mock_meta(),
        ),
        pytest.raises(ValueError, match="unique values"),
    ):
        from_aniframe_file(path)


@pytest.mark.parametrize("polar_col", ["rho", "phi", "theta"])
def test_from_aniframe_file_polar_coords_raise(tmp_path, polar_col):
    """from_aniframe_file raises ValueError for polar/spherical columns."""
    df = _minimal_df()
    df[polar_col] = 0.0
    path = _make_parquet(tmp_path, df=df)
    with (
        patch(
            "movement.io.load_aniframe._decode_aniframe_metadata",
            return_value=_mock_meta(),
        ),
        pytest.raises(ValueError, match="Polar"),
    ):
        from_aniframe_file(path)


def test_from_aniframe_file_bottom_left_warns(tmp_path):
    """from_aniframe_file warns when point_of_reference is 'bottom_left'."""
    path = _make_parquet(tmp_path)
    with (
        patch(
            "movement.io.load_aniframe._decode_aniframe_metadata",
            return_value=_mock_meta(point_of_reference="bottom_left"),
        ),
        pytest.warns(UserWarning, match="bottom-left"),
    ):
        from_aniframe_file(path)


def test_from_aniframe_file_extra_attrs_attached(aniframe_parquet):
    """from_aniframe_file attaches space_unit and reference_frame attrs."""
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(
            unit_space="mm",
            reference_frame="egocentric",
        ),
    ):
        ds = from_aniframe_file(aniframe_parquet)

    assert ds.attrs.get("space_unit") == "mm"
    assert ds.attrs.get("reference_frame") == "egocentric"


# ---------------------------------------------------------------------------
# Auto-inference (load_dataset source_software="auto")
# ---------------------------------------------------------------------------


def test_auto_inference_detects_aniframe_parquet(tmp_path):
    """load_dataset with source_software='auto' detects aniframe parquet."""
    from movement.io import load_dataset

    path = _make_parquet(tmp_path)
    with patch(
        "movement.io.load_aniframe._decode_aniframe_metadata",
        return_value=_mock_meta(),
    ):
        ds = load_dataset(path, source_software="auto")

    assert isinstance(ds, xr.Dataset)


def test_auto_inference_rejects_plain_parquet(tmp_path):
    """load_dataset auto-inference rejects non-aniframe parquet."""
    from movement.io import load_dataset

    path = _make_parquet(tmp_path, r_meta_bytes=None)
    with pytest.raises(ValueError, match="Could not infer source_software"):
        load_dataset(path, source_software="auto")
